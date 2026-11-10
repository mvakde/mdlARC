"""Coincident switches: identity-matched placement pulses toggle registered motifs.

Standard library + NumPy; no parent imports, files, hidden dictionaries or RNG in
solve. Generation renders from sampled normalized designs; solving independently
recovers the designs and pulse counts from the finished input.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= s <= 30 for s in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.tobytes()


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _bands(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(a[0]), int(a[-1]) + 1) for a in np.split(indices, cuts)]


def _read_input(input_grid):
    """Infer both frame systems, source identities, registration and overlaps."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), "missing background margin"
    bands = _bands(np.any(g != bg, axis=1))
    assert len(bands) == 2, "expected a library above a pulse-sheet band"
    rectangles = []
    for top, bottom in bands:
        columns = np.flatnonzero(np.any(g[top:bottom] != bg, axis=0))
        rectangles.append(g[top:bottom, columns[0]:columns[-1] + 1])
    library, sheets = rectangles
    sep = int(library[0, 0])
    assert sep != bg and sheets[0, 0] == sep
    for rect in rectangles:
        assert all(np.all(edge == sep) for edge in (rect[0], rect[-1], rect[:, 0], rect[:, -1])), "broken frame"
    rows = np.flatnonzero(np.all(library == sep, axis=1))
    cols = np.flatnonzero(np.all(library == sep, axis=0))
    assert 2 <= len(rows) <= 4 and len(cols) == 3
    assert len(set(np.diff(rows))) == len(set(np.diff(cols))) == 1
    th, tw = int(rows[1] - rows[0] - 1), int(cols[1] - cols[0] - 1)
    assert 3 <= th <= 5 and 3 <= tw <= 5
    assert rows[0] == cols[0] == 0 and rows[-1] == library.shape[0] - 1
    assert cols[-1] == library.shape[1] - 1

    scols = np.flatnonzero(np.all(sheets == sep, axis=0))
    srows = np.flatnonzero(np.all(sheets == sep, axis=1))
    assert len(scols) >= 2 and len(set(np.diff(scols))) == 1
    assert srows.tolist() == [0, 2, sheets.shape[0] - 1]
    mh, mw = sheets.shape[0] - 4, int(scols[1] - scols[0] - 1)
    assert mh >= 1 and mw >= 1 and mh * th <= 30 and mw * tw <= 30
    labels, layers, headers = set(), [], []
    for left in scols[:-1]:
        header = sheets[1, left + 1:left + 1 + mw]
        label = int(header[0])
        assert label not in (bg, sep) and np.all(header[1:] == bg)
        panel = sheets[3:3 + mh, left + 1:left + 1 + mw]
        assert np.all((panel == bg) | (panel == label))
        labels.add(label)
        headers.append(label)
        layers.append(panel == label)

    templates, raw, row_order = {}, {}, []
    for top in rows[:-1]:
        pair = [library[top + 1:top + 1 + th, c + 1:c + 1 + tw].copy()
                for c in cols[:-1]]
        present = set(int(v) for p in pair for v in np.unique(p))
        identity = present & labels
        assert len(identity) == 1, "a reference row must identify one material"
        label = identity.pop()
        assert label not in templates and sep not in present
        maps, carrier_shapes = [], []
        for panel in pair:
            cy, cx = np.nonzero(panel == label)
            assert len(cy) >= 3, "scaffold needs at least three cells"
            ay, ax = int(cy.min()), int(cx.min())
            carrier_shapes.append({(int(y) - ay, int(x) - ax) for y, x in zip(cy, cx)})
            maps.append({(int(y) - ay, int(x) - ax): int(panel[y, x])
                         for y, x in zip(*np.nonzero(panel != bg))})
        assert carrier_shapes[0] == carrier_shapes[1], "scaffold shapes disagree"
        union = set(maps[0]) | set(maps[1])
        y0, y1, x0, x1 = _bounds(union)
        assert (y1 - y0 + 1, x1 - x0 + 1) == (th, tw), "registered union must fill the tile's extent"
        normalized = []
        for mapping in maps:
            tile = np.full((th, tw), bg, dtype=np.int8)
            for (y, x), color in mapping.items():
                tile[y - y0, x - x0] = color
            normalized.append(tile)
        a, b = normalized
        assert not np.array_equal(a, b), "identical switch states"
        assert np.array_equal(a == label, b == label)
        templates[label] = tuple(normalized)
        raw[label] = tuple(pair)
        row_order.append(label)
    assert set(templates) == labels, "unmatched sheet or reference identity"
    counts = {label: np.zeros((mh, mw), dtype=np.int16) for label in labels}
    for label, layer in zip(headers, layers):
        counts[label] += layer
    assert all(np.any(c) for c in counts.values()), "unused material"
    active = sum((c > 0).astype(np.int8) for c in counts.values())
    assert np.all(active <= 1), "different materials may not share a macro-site"
    return dict(bg=bg, sep=sep, th=th, tw=tw, mh=mh, mw=mw,
                templates=templates, raw=raw, counts=counts, headers=headers,
                layers=layers, row_order=row_order, bands=bands,
                sheet_columns=scols.tolist())


def _render(info, shortcut=None):
    """Blockwise input-derived answer, with explicit diagnostic alternatives."""
    out = np.full((info['mh'] * info['th'], info['mw'] * info['tw']),
                  info['bg'], dtype=np.int8)
    first = info['row_order'][0]
    for label, counts in info['counts'].items():
        for y, x in zip(*np.nonzero(counts)):
            n = int(counts[y, x])
            if shortcut == 'erase_even' and n % 2 == 0:
                continue
            state = 0 if shortcut == 'always_left' else 1 if shortcut == 'always_right' else n % 2
            material = first if shortcut == 'ignore_identity' else label
            tile = (info['raw'] if shortcut == 'box_alignment' else info['templates'])[material][state]
            th, tw = info['th'], info['tw']
            out[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input using only its visible references, headers and pulses."""
    return _render(_read_input(input_grid))


def _footprint(rng, h, w):
    y, x = np.indices((h, w))
    style = int(rng.integers(5))
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    if style == 0:
        mask = (y == cy) | (x == cx)
    elif style == 1:
        mask = (y == 0) | (y == h - 1) | (x == 0) | (x == w - 1)
    elif style == 2:
        mask = (x == np.rint(y * (w - 1) / max(1, h - 1)))
        mask |= np.fliplr(mask)
    elif style == 3:
        mask = rng.random((h, w)) < rng.uniform(.35, .85)
    else:
        mask = ((y + x) % 2 == int(rng.integers(2)))
    # Edge witnesses make the logical dimensions explicit, without a fixed glyph.
    mask[0, int(rng.integers(w))] = True
    mask[-1, int(rng.integers(w))] = True
    mask[int(rng.integers(h)), 0] = True
    mask[int(rng.integers(h)), -1] = True
    for i in rng.permutation(h * w)[:min(3, h * w)]:
        mask.flat[int(i)] = True
    return {(int(y), int(x)) for y, x in zip(*np.nonzero(mask))}


def _design(rng, th, tw, label, accents, force_move=False):
    mode = 'combined' if force_move else str(rng.choice(['recolor', 'relocate', 'combined']))
    if th < 4 and tw < 4:
        mode = 'recolor'
    horizontal = mode != 'recolor' and (th < 4 or (tw >= 4 and rng.random() < .5))
    h, w = (tw, th) if horizontal else (th, tw)
    ch, offset = (h, 0) if mode == 'recolor' else (h - 2, 1)
    core = {(y + offset, x) for y, x in _footprint(rng, ch, w)}
    original_box = _bounds(core)
    changing = set()
    if mode != 'relocate':
        candidates = [(y, x) for y in range(offset, offset + ch) for x in range(w)]
        rng.shuffle(candidates)
        target = int(rng.integers(1, 4))
        for point in candidates:
            remaining = core - changing - {point}
            if len(remaining) >= 3 and _bounds(remaining) == original_box:
                changing.add(point)
                if len(changing) == target:
                    break
        assert changing
    scaffold = core - changing
    common = {p: label for p in scaffold}
    # Optional stable accent: it is copied in both states, like B's stable interior.
    if rng.random() < .55:
        points = sorted(scaffold)
        rng.shuffle(points)
        for point in points:
            rest = scaffold - {point}
            if len(rest) >= 3 and _bounds(rest) == original_box:
                common[point] = accents[4]
                break
    maps = []
    split_accents = rng.random() < .5
    same_endpoint_color = mode == 'relocate' and rng.random() < .35
    for state in range(2):
        mapping = dict(common)
        for point in changing:
            mapping[point] = accents[state]
        if mode != 'recolor':
            length = int(rng.integers(1, w + 1))
            start = int(rng.integers(w - length + 1))
            endpoint_color = accents[0 if same_endpoint_color else state + (2 if split_accents else 0)]
            for x in range(start, start + length):
                mapping[(0 if state == 0 else h - 1, x)] = endpoint_color
        maps.append(mapping)
    tiles = []
    # -1 is only a temporary in-memory background, never an ARC output color.
    for mapping in maps:
        tile = np.full((h, w), -1, dtype=np.int8)
        for point, color in mapping.items():
            tile[point] = color
        if horizontal:
            tile = tile.T
        tiles.append(tile.copy())
    if rng.random() < .5:
        tiles = [np.flipud(t).copy() for t in tiles]
    if rng.random() < .5:
        tiles = [np.fliplr(t).copy() for t in tiles]
    if rng.random() < .5:
        tiles.reverse()
    return tuple(tiles)


def _positions(rng, h, w):
    points = [(y, x) for y in range(h) for x in range(w)]
    style = int(rng.integers(5))
    if style == 0:
        rng.shuffle(points)
    elif style == 1:
        points.sort(key=lambda p: (p[0], p[1] if p[0] % 2 == 0 else -p[1]))
    elif style == 2:
        points.sort(key=lambda p: (sum(p), p[0]))
    elif style == 3:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        points.sort(key=lambda p: (abs(p[0] - cy) + abs(p[1] - cx), p))
    else:
        points.sort(key=lambda p: (min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]), p))
    # Reverse/mirror the map order without changing its row/column dimensions.
    fy, fx = bool(rng.integers(2)), bool(rng.integers(2))
    return [(h - 1 - y if fy else y, w - 1 - x if fx else x) for y, x in points]


def _build_input(rng, bg, sep, labels, templates, sheet_data, mh, mw, witness):
    th, tw = templates[labels[0]][0].shape
    k, count = len(labels), len(sheet_data)
    library = np.full((k * (th + 1) + 1, 2 * (tw + 1) + 1), bg, dtype=np.int8)
    library[::th + 1, :] = sep
    library[:, ::tw + 1] = sep
    for row, label in enumerate(labels):
        for state, tile in enumerate(templates[label]):
            yy, xx = np.nonzero(tile != bg)
            crop = tile[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
            dy = 0 if witness else int(rng.integers(th - crop.shape[0] + 1))
            dx = 0 if witness else int(rng.integers(tw - crop.shape[1] + 1))
            top, left = row * (th + 1) + 1 + dy, state * (tw + 1) + 1 + dx
            library[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    sheets = np.full((mh + 4, count * (mw + 1) + 1), bg, dtype=np.int8)
    sheets[[0, 2, mh + 3], :] = sep
    sheets[:, ::mw + 1] = sep
    for j, (label, mask) in enumerate(sheet_data):
        left = j * (mw + 1) + 1
        sheets[1, left] = label
        sheets[3:3 + mh, left:left + mw] = np.where(mask, label, bg)
    height, width = library.shape[0] + sheets.shape[0] + 3, max(library.shape[1], sheets.shape[1]) + 2
    assert height <= 30 and width <= 30
    extra_y = int(height < 30 and rng.random() < .35)
    extra_x = int(width < 30 and rng.random() < .35)
    g = np.full((height + extra_y, width + extra_x), bg, dtype=np.int8)
    g[1:1 + library.shape[0], 1:1 + library.shape[1]] = library
    top = library.shape[0] + 2
    g[top:top + sheets.shape[0], 1:1 + sheets.shape[1]] = sheets
    return g


def _make_pair(rng, split, index):
    witness = split == 'train' and index == 0
    k = 2 if witness else int(rng.integers(1, 4))
    primary = 4 if split == 'test' else 3 if witness else int(rng.integers(2, 4))
    repetitions = [primary] + [1] * (k - 1)
    if k > 1 and sum(repetitions) < 6 and (witness or rng.random() < .4):
        repetitions[1] += 1
    sheets_count = sum(repetitions)
    th = int(rng.integers(4 if witness else 3, 6))
    tw = int(rng.integers(3, 6))
    max_h = min(6, 30 // th, 30 - k * (th + 1) - 8)
    max_w = min(6, 30 // tw, 27 // sheets_count - 1)
    choices = [(h, w) for h in range(2, max_h + 1) for w in range(2, max_w + 1)
               if h * w >= sheets_count + 1]
    assert choices, "bounded parameter ranges must fit"
    mh, mw = choices[int(rng.integers(len(choices)))]
    palette = [int(v) for v in rng.permutation(10)]
    bg, sep = palette[:2]
    labels = palette[2:2 + k]
    accents = palette[2 + k:]
    # At least five nonstructural colors are available, even with three identities.
    templates = {}
    for i, label in enumerate(labels):
        native = _design(rng, th, tw, label, accents, force_move=witness and i == 0)
        templates[label] = tuple(np.where(t == -1, bg, t).astype(np.int8) for t in native)
    counts = {label: np.zeros((mh, mw), dtype=np.int16) for label in labels}
    order = _positions(rng, mh, mw)
    cursor = 0
    for label, n in zip(labels, repetitions):
        multiplicities = [int(v) for v in rng.permutation(np.arange(1, n + 1))]
        for multiplicity in multiplicities:
            counts[label][order[cursor]] = multiplicity
            cursor += 1
    reserve_blank = witness or split == 'test' or rng.random() < .6
    density = float(rng.choice([.15, .50, 1.0]))
    weights = rng.dirichlet(np.full(k, .8))
    for point in order[cursor:len(order) - int(reserve_blank)]:
        if rng.random() < density:
            i = int(rng.choice(k, p=weights))
            counts[labels[i]][point] = int(rng.integers(1, repetitions[i] + 1))
    sheet_data = []
    for label, n in zip(labels, repetitions):
        masks = [np.zeros((mh, mw), dtype=bool) for _ in range(n)]
        for point in zip(*np.nonzero(counts[label])):
            subset = rng.permutation(n)[:int(counts[label][point])]
            for j in subset:
                masks[int(j)][point] = True
        sheet_data.extend((label, mask) for mask in masks)
    rng.shuffle(sheet_data)
    # Make a visibly misleading first-column alignment in the first demonstration.
    if witness and sheet_data[0][0] == labels[0]:
        j = next(j for j, (label, _) in enumerate(sheet_data) if label != labels[0])
        sheet_data[0], sheet_data[j] = sheet_data[j], sheet_data[0]
    inp = _build_input(rng, bg, sep, labels, templates, sheet_data, mh, mw, witness)
    # Independent construction uses Boolean Kronecker masks, not solve or parsing.
    out = np.full((mh * th, mw * tw), bg, dtype=np.int8)
    for label in labels:
        for state in (0, 1):
            macro = (counts[label] > 0) & (counts[label] % 2 == state)
            for color in np.unique(templates[label][state]):
                if color != bg:
                    out[np.kron(macro, templates[label][state] == color)] = color
    return dict(input=inp, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(16):
                pair = _make_pair(rng, split, index)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _read_input(inp)
            assert np.array_equal(out, _render(info)), 'incorrect switch expansion'
            assert out.shape == (info['mh'] * info['th'], info['mw'] * info['tw'])
            # Check the invariant scaffold separately at every active site, and
            # check untouched sites as whole background tiles.
            occupied = np.zeros((info['mh'], info['mw']), dtype=bool)
            for label, counts in info['counts'].items():
                occupied |= counts > 0
                scaffold = info['templates'][label][0] == label
                for y, x in zip(*np.nonzero(counts)):
                    block = out[y * info['th']:(y + 1) * info['th'], x * info['tw']:(x + 1) * info['tw']]
                    assert np.all(block[scaffold] == label), 'switch moved the scaffold'
            for y, x in zip(*np.nonzero(~occupied)):
                assert np.all(out[y * info['th']:(y + 1) * info['th'], x * info['tw']:(x + 1) * info['tw']] == info['bg'])
            maximum = max(int(c.max()) for c in info['counts'].values())
            assert maximum <= (3 if split == 'train' else 4)
            if split == 'test':
                assert maximum == 4, 'missing fourth-pulse generalization'
            infos[split].append(info)
    # These are observable counterexamples, not counts of hidden sampler modes.
    assert any(any({1, 2, 3} <= set(c.flat) for c in p['counts'].values()) for p in infos['train'])
    for shortcut in ('always_left', 'always_right', 'erase_even', 'ignore_identity', 'box_alignment'):
        assert any(not np.array_equal(_render(p), _render(p, shortcut)) for p in infos['train']), shortcut


def self_test():
    """Streaming, bounded checks over 42 seeds; no diagnostic files are written."""
    shapes, motifs, layouts, palettes, schedules, fingerprints = (set() for _ in range(6))
    changes, multiplicities, tile_shapes, materials = set(), set(), set(), set()
    shifted, pairs, edges, stable_accents = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        digest = hashlib.sha256()
        local_motifs = set()
        for split in ('train', 'test'):
            for pair, repeat in zip(puzzle[split], again[split]):
                for role in ('input', 'output'):
                    assert np.array_equal(pair[role], repeat[role])
                    digest.update(bytes(pair[role].shape) + pair[role].tobytes())
                p = _read_input(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                tile_shapes.add((p['th'], p['tw']))
                materials.add(len(p['templates']))
                palettes.add((p['bg'], p['sep'], tuple(p['row_order'])))
                schedules.add(tuple(sorted(p['headers'].count(c) for c in p['templates'])))
                normalized_counts = np.zeros((p['mh'], p['mw']), dtype=np.int16)
                for i, (label, count) in enumerate(p['counts'].items()):
                    normalized_counts[count > 0] = 10 * (i + 1) + count[count > 0]
                    multiplicities.update(int(c) for c in count.flat)
                layouts.add(_key(normalized_counts))
                for label, (a, b) in p['templates'].items():
                    # Preserve color relationships but remove literal palette names.
                    roles = {p['bg']: 0, label: 1}
                    encoded = tuple(tuple(roles.setdefault(int(v), len(roles)) for v in t.flat) for t in (a, b))
                    motif = (a.shape, encoded)
                    motifs.add(motif)
                    local_motifs.add(motif)
                    relocation = not np.array_equal(a != p['bg'], b != p['bg'])
                    recolor = bool(np.any((a != p['bg']) & (b != p['bg']) & (a != b)))
                    changes.add((recolor, relocation))
                    shifted += any(not np.array_equal(t, raw) for t, raw in zip((a, b), p['raw'][label]))
                    stable_accents += bool(np.any((a == b) & (a != p['bg']) & (a != label)))
                    edges += any(np.any(t[0] != p['bg']) and np.any(t[-1] != p['bg']) for t in (a, b))
                pairs += 1
        assert len(local_motifs) >= 3, 'within-puzzle geometry/state diversity'
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(shapes) >= 60
    assert len(motifs) >= 100 and len(layouts) >= 150 and len(palettes) >= 100
    assert materials == {1, 2, 3} and len(tile_shapes) >= 6
    assert multiplicities == {0, 1, 2, 3, 4}
    assert changes == {(True, False), (False, True), (True, True)}
    assert shifted >= 50 and stable_accents >= 50 and edges >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(42000 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    p = generate_puzzle(17, 2, 1)
    info = _read_input(p['test'][0]['input'])
    # A full color permutation changes every semantic role coherently.
    permutation = np.random.default_rng(222).permutation(10)
    assert np.array_equal(solve(permutation[p['test'][0]['input']]), permutation[p['test'][0]['output']])
    # Reordering complete pulse cards (including headers) cannot affect the result.
    g = p['test'][0]['input'].copy()
    top, bottom = info['bands'][1]
    left = int(np.flatnonzero(np.any(g[top:bottom] != info['bg'], axis=0))[0])
    step = info['mw'] + 1
    band = g[top:bottom].copy()
    n = len(info['headers'])
    for i in range(n):
        g[top:bottom, left + i * step + 1:left + i * step + step] = band[:, left + (n - 1 - i) * step + 1:left + (n - 1 - i) * step + step]
    assert np.array_equal(solve(g), p['test'][0]['output'])
    # Reject a wrong output, a scaffold damaged in one state, and mixed-material overlap.
    corrupted = {s: [{r: v.copy() for r, v in pair.items()} for pair in p[s]] for s in p}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad output accepted')
    original = generate_puzzle(8, 2, 1)['train'][0]['input']
    p0 = _read_input(original)
    damaged = original.copy()
    y, x = np.argwhere(damaged[1:1 + p0['th'], 1:1 + p0['tw']] == p0['row_order'][0])[0]
    damaged[1 + y, 1 + x] = p0['bg']
    top, bottom = p0['bands'][1]
    pulse_left = int(np.flatnonzero(np.any(original[top:bottom] != p0['bg'], axis=0))[0])
    overlap = original.copy()
    j = next(j for j, label in enumerate(p0['headers']) if label != p0['headers'][0])
    site = tuple(np.argwhere(p0['layers'][0])[0])
    overlap[top + 3 + site[0], pulse_left + j * (p0['mw'] + 1) + 1 + site[1]] = p0['headers'][j]
    for bad in (damaged, overlap):
        try:
            solve(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('invalid reference/overlap accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints),
                shape_pairs=len(shapes), normalized_state_pairs=len(motifs),
                pulse_layouts=len(layouts), palettes=len(palettes),
                tile_shapes=sorted(tile_shapes), material_counts=sorted(materials),
                layer_schedules=sorted(schedules), shifted_reference_pairs=shifted,
                stable_accent_pairs=stable_accents)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
