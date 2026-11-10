"""Layer peeling: explicit punch motifs subtract local rectangle-stack depth.

Standard library + NumPy, no file I/O, no hidden codebook, bounded generation.
The header and separator survive; the field is reconstructed from instructions.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _corners(bounds, diagonal):
    r0, c0, r1, c1 = bounds
    if diagonal == 0:
        return {(r0, c0), (r0, c0+1), (r0+1, c0),
                (r1, c1), (r1, c1-1), (r1-1, c1)}
    return {(r0, c1), (r0, c1-1), (r0+1, c1),
            (r1, c0), (r1, c0+1), (r1-1, c0)}


def _header(width, bg, divider, layers, book):
    height = max(len(layers)+2, max(m.shape[0]+2 for m in book.values()))
    a = np.full((height+1, width), bg, dtype=np.int8)
    for i, layer in enumerate(layers):
        a[i+1, 1:3] = layer['cue'], layer['fill']
    c = 5
    for code, mask in book.items():
        h, w = mask.shape
        a[:h+2, c:c+w+2] = divider
        a[1:h+1, c+1:c+w+1] = np.where(mask, code, bg)
        c += w+3
    a[-1] = divider
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    separators = [r for r in range(1, len(a))
                  if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    if len(separators) != 1:
        raise ValueError("one horizontal header separator is required")
    cut = separators[0]
    divider = int(a[cut, 0])
    head, field = a[:cut+1], a[cut+1:]
    if not field.size or a.shape[1] < 10:
        raise ValueError("missing work field")
    entries = []
    r = 1
    while r < cut and head[r, 1] != bg:
        entries.append((int(head[r, 1]), int(head[r, 2])))
        r += 1
    if not 1 <= len(entries) <= 4 or len({q[0] for q in entries}) != len(entries):
        raise ValueError("invalid sheet table")
    # Frames start on header row zero, after the two-column table.
    book, c = {}, 5
    while c < a.shape[1] and head[0, c] == divider:
        end = c
        while end < a.shape[1] and head[0, end] == divider:
            end += 1
        bottom = 0
        while bottom < cut and head[bottom, c] == divider:
            bottom += 1
        h, w = bottom-2, end-c-2
        if not 2 <= h <= 5 or not 2 <= w <= 5:
            raise ValueError("invalid motif frame")
        interior = head[1:bottom-1, c+1:end-1]
        colors = set(map(int, np.unique(interior))) - {bg}
        if len(colors) != 1:
            raise ValueError("each motif has one code color")
        code = colors.pop()
        mask = interior == code
        if code in book or not all((mask[0].any(), mask[-1].any(),
                                    mask[:, 0].any(), mask[:, -1].any())):
            raise ValueError("motif codes must be unique and bounds tight")
        if mask.all() or not mask.any():
            raise ValueError("a motif must have both solid and blank cells")
        book[code] = mask.copy()
        c = end+1
    if not 1 <= len(book) <= 2:
        raise ValueError("one or two motifs required")
    cues, paints, codes = {q[0] for q in entries}, {q[1] for q in entries}, set(book)
    roles = [{bg}, {divider}, cues, paints, codes]
    if any(roles[i] & roles[j] for i in range(5) for j in range(i)):
        raise ValueError("semantic palette roles collide")
    if not set(map(int, np.unique(field))) <= {bg} | cues | codes:
        raise ValueError("unexplained field material")
    layers = []
    for cue, fill in entries:
        cells = {tuple(map(int, q)) for q in np.argwhere(field == cue)}
        if len(cells) != 6:
            raise ValueError("each sheet needs two intact L triplets")
        rr, cc = zip(*cells)
        bounds = (min(rr), min(cc), max(rr), max(cc))
        h, w = bounds[2]-bounds[0]+1, bounds[3]-bounds[1]+1
        if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
            raise ValueError("degenerate or ambiguous corner pair")
        diagonals = [d for d in (0, 1) if _corners(bounds, d) == cells]
        if len(diagonals) != 1:
            raise ValueError("not opposite L corners")
        layers.append(dict(cue=cue, fill=fill, bounds=bounds,
                           diagonal=diagonals[0], markers=cells))
    if not np.array_equal(_header(a.shape[1], bg, divider, layers, book), head):
        raise ValueError("noncanonical or damaged legend")
    anchors = [(int(r), int(c), int(field[r, c]))
               for r, c in np.argwhere(np.isin(field, list(book)))]
    if not anchors:
        raise ValueError("at least one punch instruction required")
    return a, cut, bg, divider, field, layers, book, anchors


def _hits(shape, anchors, book, windows=False):
    hits = np.zeros(shape, dtype=np.int16)
    for r, c, code in anchors:
        mask = book[code]
        h, w = min(mask.shape[0], shape[0]-r), min(mask.shape[1], shape[1]-c)
        hits[r:r+h, c:c+w] += 1 if windows else mask[:h, :w]
    return hits


def _surface(shape, bg, layers, hits):
    """Array-derived solution: skip k covering table rows at each pixel."""
    out = np.full(shape, bg, dtype=np.int8)
    pending, assigned = hits.copy(), np.zeros(shape, dtype=bool)
    for layer in layers:
        r0, c0, r1, c1 = layer['bounds']
        sl = np.s_[r0:r1+1, c0:c1+1]
        live = ~assigned[sl]
        remove = live & (pending[sl] > 0)
        keep = live & ~remove
        out[sl][keep] = layer['fill']
        for r, c in layer['markers']:
            if keep[r-r0, c-c0]:
                out[r, c] = layer['cue']
        pending[sl][remove] -= 1
        assigned[sl][keep] = True
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, examples, files, generation state or learned latent vocabulary."""
    a, cut, bg, _, field, layers, book, anchors = _parse(input_grid)
    a[cut+1:] = _surface(field.shape, bg, layers, _hits(field.shape, anchors, book))
    return a


def _construct_output(shape, bg, layers, book, anchors):
    """Generation uses literal sequential removal, rather than solver counts."""
    alive = np.zeros((len(layers), *shape), dtype=bool)
    colors = np.full(alive.shape, bg, dtype=np.int8)
    for i, layer in enumerate(layers):
        r0, c0, r1, c1 = layer['bounds']
        alive[i, r0:r1+1, c0:c1+1] = True
        colors[i, r0:r1+1, c0:c1+1] = layer['fill']
        for r, c in layer['markers']:
            colors[i, r, c] = layer['cue']
    for r, c, code in anchors:
        for dr, dc in np.argwhere(book[code]):
            rr, cc = r+int(dr), c+int(dc)
            if rr < shape[0] and cc < shape[1]:
                stack = np.flatnonzero(alive[:, rr, cc])
                if len(stack):
                    alive[int(stack[0]), rr, cc] = False
    out = np.full(shape, bg, dtype=np.int8)
    for i in reversed(range(len(layers))):
        out[alive[i]] = colors[i][alive[i]]
    return out


def _depth(shape, layers):
    d = np.zeros(shape, dtype=np.int8)
    for layer in layers:
        r0, c0, r1, c1 = layer['bounds']
        d[r0:r1+1, c0:c1+1] += 1
    return d


def _features(field, bg, layers, book, anchors, output):
    shape = field.shape
    depth = _depth(shape, layers)
    hits = _hits(shape, anchors, book)
    base = _surface(shape, bg, layers, np.zeros(shape, dtype=np.int16))
    union = _surface(shape, bg, layers, np.minimum(hits, 1))
    flattened = np.where(hits > 0, bg, base)
    probes = set()
    bottom = right = False
    for r, c, code in anchors:
        mask = book[code]
        h, w = mask.shape
        hh, ww = min(h, shape[0]-r), min(w, shape[1]-c)
        sl = np.s_[r:r+hh, c:c+ww]
        if hh == h and ww == w:
            if (np.all(depth[sl] == 1) and len(np.unique(base[sl])) == 1
                    and base[r, c] != bg and np.array_equal(hits[sl], mask)
                    and np.array_equal(output[sl], np.where(mask, bg, base[sl]))):
                probes.add(code)
        elif (h > hh or w > ww):
            without = hits.copy()
            without[sl] -= mask[:hh, :ww]
            if np.any(_surface(shape, bg, layers, without) != output):
                bottom |= hh < h
                right |= ww < w
    cue_cells = np.zeros(shape, dtype=bool)
    for layer in layers:
        for q in layer['markers']:
            cue_cells[q] = base[q] == layer['cue']
    return dict(probes=probes,
                reveal=bool(np.any((hits == 1) & (depth >= 2) & (output != bg) & (output != base))),
                multiplicity=bool(np.any(output != union)),
                third=bool(np.any((hits == 2) & (depth >= 3) & (output != bg) & (output != union))),
                priority=bool(np.any(output != _surface(shape, bg, list(reversed(layers)), hits))),
                no_a=bool(np.any(output != flattened)),
                no_b=bool(np.any(output != base)),
                window=bool(np.any(output != _surface(shape, bg, layers, _hits(shape, anchors, book, True)))),
                marker=bool(np.any(cue_cells & (hits > 0) & (output != base))),
                bottom=bool(bottom), right=bool(right),
                max_depth=int(depth.max()), max_hits=int(hits.max()),
                anchor_count=len(anchors), sheet_count=len(layers))


def _sample_mask(rng):
    for _ in range(100):
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        m = np.zeros((h, w), dtype=bool)
        kind = int(rng.integers(5))
        if kind == 0:
            m[0] = m[-1] = True
            m[:, 0] = m[:, -1] = True
        elif kind == 1:
            m[:, 0] = m[-1] = True
            m[int(rng.integers(h-1)), :int(rng.integers(2, w+1))] = True
        elif kind == 2:
            m[h//2] = True
            m[:, w//2] = True
            m[0, -1] = True
        elif kind == 3:
            r = c = 0
            m[r, c] = True
            while r < h-1 or c < w-1:
                if r == h-1 or (c < w-1 and rng.random() < .5):
                    c += 1
                else:
                    r += 1
                m[r, c] = True
        else:
            m = rng.random((h, w)) < .65
        if rng.random() < .5:
            m = m[::-1]
        if rng.random() < .5:
            m = m[:, ::-1]
        if rng.random() < .5:
            m = m.T
        if (all((m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any()))
                and 3 <= m.sum() < m.size):
            return m.copy()
    raise RuntimeError("bounded mask sampling failed")


def _box(rng, H, W, thin=False):
    h = int(rng.integers(2 if thin else 5, H+1))
    w = int(rng.integers(2 if thin else 5, W+1))
    if min(h, w) == 2 and max(h, w) < 4:
        w = 4
    r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
    return r, c, r+h-1, c+w-1


def _geometry(rng, H, W, n, mode, edges):
    boxes = []
    for i in range(n):
        if mode == 'nested' and i > 0:
            r0, c0, r1, c1 = boxes[-1]
            if r1-r0 >= 6 and c1-c0 >= 6:
                b = (r0+1, c0+1, r1-1, c1-1)
            else:
                b = _box(rng, H, W)
        elif mode == 'bands' and i < 2:
            if i == 0:
                h = int(rng.integers(4, max(5, H//2+1)))
                r = int(rng.integers(H-h+1))
                b = (r, int(rng.integers(3)), r+h-1, W-1-int(rng.integers(3)))
            else:
                w = int(rng.integers(4, max(5, W//2+1)))
                c = int(rng.integers(W-w+1))
                b = (int(rng.integers(3)), c, H-1-int(rng.integers(3)), c+w-1)
        else:
            b = _box(rng, H, W, thin=i >= 2 and rng.random() < .35)
        if edges and i == 0:
            b = (b[0], b[1], H-1, b[3])
        if edges and i == 1:
            b = (b[0], b[1], b[2], W-1)
        boxes.append(b)
    if len(set(boxes)) != n:
        return None
    layers, used = [], set()
    for b in boxes:
        diagonal = int(rng.integers(2))
        markers = _corners(b, diagonal)
        if used & markers or len(markers) != 6:
            return None
        used.update(markers)
        layers.append(dict(bounds=b, diagonal=diagonal, markers=markers))
    return [layers[int(i)] for i in rng.permutation(n)]


def _choose_anchors(rng, field, bg, layers, book, required, clipping):
    H, W = field.shape
    depth = _depth(field.shape, layers)
    base = _surface(field.shape, bg, layers, np.zeros(field.shape, dtype=np.int16))
    anchors, positions = [], set()
    hits = np.zeros(field.shape, dtype=np.int16)
    reserved = np.zeros(field.shape, dtype=bool)
    offsets = {c: [tuple(map(int, q)) for q in np.argwhere(m)] for c, m in book.items()}

    def eligible(r, c, code, allow_clip=False):
        h, w = book[code].shape
        if not (0 <= r < H and 0 <= c < W) or field[r, c] != bg or (r, c) in positions:
            return None
        if not allow_clip and (r+h > H or c+w > W):
            return None
        if reserved[r, c]:
            return None
        cells = [(r+dr, c+dc) for dr, dc in offsets[code] if r+dr < H and c+dc < W]
        if not cells or any(reserved[q] for q in cells):
            return None
        return cells

    def add(r, c, code, allow_clip=False):
        cells = eligible(r, c, code, allow_clip)
        if cells is None:
            return False
        anchors.append((r, c, code))
        positions.add((r, c))
        for q in cells:
            hits[q] += 1
        return True

    if 'probes' in required:
        for code, mask in book.items():
            h, w = mask.shape
            options = []
            for r in range(H-h+1):
                for c in range(W-w+1):
                    sl = np.s_[r:r+h, c:c+w]
                    if (not reserved[sl].any() and np.all(depth[sl] == 1)
                            and np.all(base[sl] == base[r, c]) and base[r, c] != bg
                            and eligible(r, c, code) is not None):
                        options.append((r, c))
            if not options:
                return None
            r, c = options[int(rng.integers(len(options)))]
            add(r, c, code)
            reserved[r:r+h, c:c+w] = True

    needed = 2 if required & {'multiplicity', 'third'} else 1
    if required & {'reveal', 'multiplicity', 'third'}:
        targets = np.argwhere(depth >= (3 if 'third' in required else 2))
        found = False
        for ix in rng.permutation(len(targets)):
            tr, tc = map(int, targets[int(ix)])
            if hits[tr, tc] or reserved[tr, tc]:
                continue
            # The mandatory depth witness must reveal interior paint, rather
            # than relying on a coincident printed corner to distinguish ranks.
            if any((tr, tc) in o['markers'] for o in layers):
                continue
            local = [o for o in layers if o['bounds'][0] <= tr <= o['bounds'][2]
                     and o['bounds'][1] <= tc <= o['bounds'][3]]
            vals = [o['cue'] if (tr, tc) in o['markers'] else o['fill'] for o in local]
            if 'third' in required and vals[1] == vals[2]:
                continue
            options = []
            for code in book:
                for dr, dc in offsets[code]:
                    r, c = tr-dr, tc-dc
                    if eligible(r, c, code) is not None:
                        options.append((r, c, code))
            options = [options[int(i)] for i in rng.permutation(len(options))]
            chosen = []
            for q in options:
                if q[:2] not in [z[:2] for z in chosen]:
                    chosen.append(q)
                if len(chosen) == needed:
                    break
            if len(chosen) < needed:
                continue
            for q in chosen:
                add(*q)
            found = True
            break
        if not found:
            return None

    if 'marker' in required:
        targets = [q for o in layers for q in sorted(o['markers'])
                   if base[q] == o['cue'] and hits[q] == 0]
        found = False
        for ix in rng.permutation(len(targets)):
            tr, tc = targets[int(ix)]
            for code in book:
                for dr, dc in offsets[code]:
                    if add(tr-dr, tc-dc, code):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None

    if 'clip' in required:
        for axis in (0, 1):
            options = []
            for code, mask in book.items():
                h, w = mask.shape
                for r in range(max(0, H-h+1) if axis == 0 else 0, H):
                    for c in range(max(0, W-w+1) if axis == 1 else 0, W):
                        cells = eligible(r, c, code, True)
                        if cells and any(depth[q] > hits[q] for q in cells):
                            options.append((r, c, code))
            if not options:
                return None
            add(*options[int(rng.integers(len(options)))], allow_clip=True)

    for _ in range(int(rng.integers(0, 3)) + int(not anchors)):
        options = []
        for code in book:
            for _ in range(30):
                r, c = int(rng.integers(H)), int(rng.integers(W))
                cells = eligible(r, c, code, clipping)
                if cells and any(depth[q] > hits[q] for q in cells):
                    options.append((r, c, code))
        if options:
            add(*options[int(rng.integers(len(options)))], allow_clip=clipping)
    return anchors or None


def _make_pair(rng, masks, n, required, clipping):
    min_width = max(15, 5+sum(m.shape[1]+3 for m in masks))
    for _ in range(96):
        head_h = max(n+2, max(m.shape[0]+2 for m in masks))
        H, W = int(rng.integers(12, 30-head_h)), int(rng.integers(min_width, 31))
        mode = str(rng.choice(['free', 'free', 'nested', 'bands']))
        layers = _geometry(rng, H, W, n, mode, 'clip' in required)
        if layers is None:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, divider = palette[:2]
        codes = palette[2+n:2+n+len(masks)]
        book = {c: m.copy() for c, m in zip(codes, masks)}
        ink_count = min(n, int(rng.integers(2, 4)), 8-n-len(masks)) if n > 1 else 1
        inks = palette[2+n+len(masks):2+n+len(masks)+ink_count]
        fills = inks + [int(rng.choice(inks)) for _ in range(n-ink_count)]
        rng.shuffle(fills)
        field = np.full((H, W), bg, dtype=np.int8)
        for o, cue, fill in zip(layers, palette[2:2+n], fills):
            o.update(cue=cue, fill=fill)
            for q in o['markers']:
                field[q] = cue
        anchors = _choose_anchors(rng, field, bg, layers, book, required, clipping)
        if anchors is None:
            continue
        out_field = _construct_output(field.shape, bg, layers, book, anchors)
        features = _features(field, bg, layers, book, anchors, out_field)
        if 'probes' in required and features['probes'] != set(book):
            continue
        if 'clip' in required and not (features['bottom'] and features['right']):
            continue
        if any(not features[k] for k in required - {'probes', 'clip'}):
            continue
        for r, c, code in anchors:
            field[r, c] = code
        head = _header(W, bg, divider, layers, book)
        inp, out = np.vstack((head, field)), np.vstack((head, out_field))
        if not np.array_equal(solve(inp), out):
            raise AssertionError("construction and input-only solution disagree")
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded spatial sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    masks = [_sample_mask(rng) for _ in range(int(rng.integers(1, 3)))]
    if len(masks) == 2 and np.array_equal(masks[0], masks[1]):
        for _ in range(100):
            masks[1] = _sample_mask(rng)
            if not np.array_equal(masks[0], masks[1]):
                break
        else:
            raise RuntimeError("bounded distinct-motif sampling failed")
    clipping = bool(rng.integers(2))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    requirements = [set() for _ in range(nt)]
    requirements[slots[0]].update({'probes', 'reveal', 'priority', 'no_a', 'no_b', 'window'})
    requirements[slots[1]].update({'multiplicity', 'marker', 'priority', 'no_b'})
    if clipping:
        requirements[slots[1]].add('clip')
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if split == 'train':
                required = requirements[i]
                n = 2 if required else int(rng.integers(1, 3))
                allow_clip = clipping and 'clip' in required
            else:
                required = {'third', 'no_a', 'no_b', 'multiplicity'} if i == 0 else {'no_b'}
                required = required | ({'clip'} if clipping and rng.random() < .65 else set())
                n = int(rng.integers(3, 5))
                allow_clip = clipping and 'clip' in required
            for _ in range(12):
                pair = _make_pair(rng, masks, n, required, allow_clip)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _motif_signature(book):
    return tuple(sorted((m.shape, m.tobytes()) for m in book.values()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, vocabulary = set(), None
    observed, trained_probes = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            _, cut, bg, _, field, layers, book, anchors = _parse(inp)
            assert np.array_equal(out, solve(inp)), "incorrect output from unchanged input"
            assert np.array_equal(out[:cut+1], inp[:cut+1]), "legend changed"
            signature = _motif_signature(book)
            if vocabulary is None:
                vocabulary = signature
            assert signature == vocabulary, "motif vocabulary changed inside puzzle"
            expected = _construct_output(field.shape, bg, layers, book, anchors)
            assert np.array_equal(expected, out[cut+1:]), "sequential peeling disagreement"
            f = _features(field, bg, layers, book, anchors, out[cut+1:])
            assert f['no_b'], "punches have no effect"
            if split == 'train':
                for code in f['probes']:
                    trained_probes.add((book[code].shape, book[code].tobytes()))
            observed[split].append(f)
    assert trained_probes == set(vocabulary), "some motif lacks a complete visible impression"
    for key in ('reveal', 'multiplicity', 'priority', 'no_a', 'no_b', 'window', 'marker'):
        assert any(f[key] for f in observed['train']), "missing training witness: " + key
    assert any(f['third'] for f in observed['test']), "missing deeper-stack test shift"
    for key in ('bottom', 'right'):
        assert not any(f[key] for f in observed['test']) or any(f[key] for f in observed['train']), "untaught clipping"
    assert max(f['sheet_count'] for f in observed['train']) <= 2
    assert min(f['sheet_count'] for f in observed['test']) >= 3
    return True


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in a)


def _rule_regressions():
    """Small explicit expectations for local ranks, printed marks and clipping."""
    shape, bg = (12, 16), 0
    layers = []
    for cue, fill, bounds, diagonal in ((2, 5, (0, 0, 9, 10), 0),
                                        (3, 6, (2, 2, 11, 13), 1),
                                        (4, 7, (1, 5, 10, 15), 0)):
        layers.append(dict(cue=cue, fill=fill, bounds=bounds, diagonal=diagonal,
                           markers=_corners(bounds, diagonal)))
    for k, expected in enumerate((5, 6, 7, 0, 0)):
        hits = np.zeros(shape, dtype=np.int16)
        hits[5, 6] = k
        assert _surface(shape, bg, layers, hits)[5, 6] == expected
    hits = np.zeros(shape, dtype=np.int16)
    hits[1, 7] = 1  # The second table row is absent here; reveal the third.
    hits[0, 0] = 1  # Printed corner removed together with its sole sheet.
    hits[2, 5] = 2  # Third sheet's own printed corner becomes visible.
    out = _surface(shape, bg, layers, hits)
    assert out[1, 7] == 7 and out[0, 0] == 0 and out[2, 5] == 4
    book = {8: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)}
    anchors = [(5, 6, 8), (5, 7, 8), (6, 6, 8), (11, 14, 8)]
    field = np.full(shape, bg, dtype=np.int8)
    for layer in layers:
        for q in layer['markers']:
            field[q] = layer['cue']
    for r, c, code in anchors:
        assert field[r, c] == bg
        field[r, c] = code
    head = _header(shape[1], bg, 1, layers, book)
    result = solve(np.vstack((head, field)))[len(head):]
    assert result[5, 6] == 5  # Blank top-left motif cell leaves the top intact.
    assert result[5, 7] == 6  # One hit reveals the second sheet.
    assert result[6, 7] == 0  # Three distinct punches exhaust three sheets.
    clipped = _hits(shape, [anchors[-1]], book)
    assert clipped.sum() == 1 and clipped[11, 15] == 1
    # Reordering the instruction list never changes physical peeling.
    assert np.array_equal(_construct_output(shape, bg, layers, book, anchors),
                          _construct_output(shape, bg, layers, book, anchors[::-1]))


def self_test():
    """42 streamed seeds, semantic ablations, corruption, determinism; no writes."""
    _rule_regressions()
    shapes, motifs, layer_counts, anchors, palettes, signatures = set(), set(), set(), set(), set(), set()
    geometries, diagonals, motif_counts, dimensions = set(), set(), set(), set()
    witnesses = {k: 0 for k in ('reveal', 'multiplicity', 'priority', 'no_a', 'no_b',
                               'window', 'marker', 'bottom', 'right', 'third')}
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                _, cut, bg, div, field, layers, book, points = _parse(inp)
                f = _features(field, bg, layers, book, points, out[cut+1:])
                pairs += 1
                shapes.add(inp.shape)
                palettes.add((bg, div, tuple(o['cue'] for o in layers), tuple(book)))
                motifs.update((m.shape, m.tobytes()) for m in book.values())
                motif_counts.add(len(book))
                layer_counts.add(len(layers))
                anchors.add(len(points))
                geometries.add(tuple(o['bounds'] for o in layers))
                diagonals.update(o['diagonal'] for o in layers)
                dimensions.update((o['bounds'][2]-o['bounds'][0]+1,
                                   o['bounds'][3]-o['bounds'][1]+1) for o in layers)
                for k in witnesses:
                    witnesses[k] += int(f[k])
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 30 and len(motifs) >= 15
    assert layer_counts == {1, 2, 3, 4} and motif_counts == {1, 2}
    assert len(anchors) >= 4 and len(geometries) >= 150 and len(dimensions) >= 40
    assert len(palettes) >= 150 and diagonals == {0, 1}
    assert all(v >= 3 for v in witnesses.values()), witnesses
    assert _same(generate_puzzle(271828, 2, 1), generate_puzzle(271828, 2, 1))
    for nt, nq in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(314159, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(98765, 2, 1)
    out = p['test'][0]['output']
    _, cut, *_ = _parse(p['test'][0]['input'])
    out[cut+1, 0] = (int(out[cut+1, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption accepted")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), motifs=len(motifs),
                geometries=len(geometries), rectangle_sizes=len(dimensions),
                anchor_counts=sorted(anchors), sheet_counts=sorted(layer_counts),
                motif_counts=sorted(motif_counts), palette_assignments=len(palettes),
                witnesses=witnesses, independent_verification=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
