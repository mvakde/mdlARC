"""Riveted mosaic: rigid docking and solid rectangular closure constrain each other.

Only NumPy and the standard library are used. No parent implementation, file,
example, seed, or generation metadata is consulted by the input-only solver.
"""
from collections import Counter
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 867508673, 64854384, 705049395, 25843041, 536085752,
              543519656, 698249331, 936324161, 864199567, 23877874,
              825187003, 317404203, 392546971, 6278342, 408262671,
              259564098, 352130903, 118148776, 248003515, 627447354,
              506072261, 262773332, 533679753, 979725362, 271037531,
              277509813, 617973047, 728412492, 203668101, 584791165,
              303328196, 6382111, 80197627, 773050783, 905801262,
              709143356, 699384088, 336542644, 720916746, 503311483,
              426514855)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        found, todo = {first}, [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _shift(cells, delta):
    dr, dc = delta
    return frozenset((r + dr, c + dc) for r, c in cells)


def _box(cells):
    cells = tuple(cells)
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normalized(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a.astype(np.int8, copy=True)


def _analyze(value):
    grid = _grid(value)
    if np.count_nonzero(grid == 9) != 1 or not np.any(grid == 0):
        raise ValueError("one visible rivet (9) and background (0) are required")
    groups = _components(map(tuple, np.argwhere(grid != 0).tolist()))
    if len(groups) < 3:
        raise ValueError("expected a core and at least two detached pieces")
    palettes = [{int(grid[q]) for q in group} - {9} for group in groups]
    common = set.intersection(*palettes)
    if len(common) != 1:
        raise ValueError("the common docking color is not identifiable")
    marker = common.pop()
    pieces, anchors = [], []
    for i, (group, palette) in enumerate(zip(groups, palettes)):
        if len(palette) != 2:
            raise ValueError("each piece has one body hue and the common marker")
        color = (palette - {marker}).pop()
        body = frozenset(q for q in group if grid[q] != marker)
        pins = frozenset(group - body)
        if len(_components(body)) != 1 or not pins:
            raise ValueError("disconnected body or missing docking cells")
        if any(grid[q] == 9 for q in body):
            anchors.append(i)
        pieces.append(dict(body=body, pins=pins, cells=frozenset(group),
                           colors={q: int(grid[q]) for q in body},
                           color=color, box=_box(group)))
    if len(anchors) != 1:
        raise ValueError("one riveted core is required")
    return grid, marker, pieces, anchors[0]


class SearchLimit(ValueError):
    pass


def _assemblies(value, rectangular=True, max_answers=2, limit=120000):
    """Enumerate connected double-owned-marker assemblies, then test closure.

    Pins are individual cells: neither local signature grouping, unique key
    areas, tree topology nor any construction-specific slot enters the search.
    Distinct colored outputs, rather than permutations, are counted.
    """
    grid, marker, pieces, anchor = _analyze(value)
    n, (height, width) = len(pieces), grid.shape
    links = {}
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces):
            if i == j:
                continue
            options = {}
            deltas = sorted({(r - s, c - t) for r, c in a['pins']
                             for s, t in b['pins']})
            for d in deltas:
                body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                if body & a['cells'] or pins & a['body']:
                    continue
                for q in a['pins'] & pins:
                    options.setdefault(q, []).append(d)
            links[i, j] = options
    answers, visited = {}, set()
    nodes = 0

    def visit(positions, bodies, singles, doubles, owners):
        nonlocal nodes
        state = tuple(sorted(positions.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise SearchLimit("complete assembly search exceeded its budget")
        if not singles:
            if len(positions) != n:
                return
            occupied = bodies | doubles
            t, l, b, r = _box(occupied)
            if rectangular and len(occupied) != (b - t + 1) * (r - l + 1):
                return
            out = np.zeros(grid.shape, dtype=np.int8)
            for i, d in positions.items():
                for (r0, c0), color in pieces[i]['colors'].items():
                    out[r0 + d[0], c0 + d[1]] = color
            for q in doubles:
                out[q] = marker
            answers.setdefault(out.tobytes(), (out, dict(positions)))
            return
        if len(positions) == n:
            return
        best = None
        for q in sorted(singles):
            i, di = owners[q], positions[owners[q]]
            local = (q[0] - di[0], q[1] - di[1])
            options = []
            for j, piece in enumerate(pieces):
                if j in positions:
                    continue
                for rel in links[i, j].get(local, ()):
                    d = (di[0] + rel[0], di[1] + rel[1])
                    t, l, b, r = piece['box']
                    if not (0 <= t + d[0] <= b + d[0] < height
                            and 0 <= l + d[1] <= r + d[1] < width):
                        continue
                    bp, bm = _shift(piece['body'], d), _shift(piece['pins'], d)
                    if bp & (bodies | singles | doubles) or bm & (bodies | doubles):
                        continue
                    options.append((j, d, bp, bm, bm & singles))
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for j, d, bp, bm, shared in best:
            new_owners = dict(owners)
            new_owners.update({q: j for q in bm - shared})
            positions[j] = d
            visit(positions, bodies | bp, (singles | bm) - shared,
                  doubles | shared, new_owners)
            del positions[j]
            if len(answers) >= max_answers:
                return

    core = pieces[anchor]
    visit({anchor: (0, 0)}, core['body'], core['pins'], frozenset(),
          {q: anchor for q in core['pins']})
    return list(answers.values()), dict(marker=marker, pieces=pieces,
                                       anchor=anchor, nodes=nodes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    answers, _ = _assemblies(input_grid)
    if len(answers) != 1:
        raise ValueError("input must have exactly one colored rectangular assembly")
    return answers[0][0].copy()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _tooth(rng, height):
    # Small ramps, teeth and steps are genuine complementary shape contours.
    kind = int(rng.integers(4))
    if kind == 0:
        return np.zeros(height, dtype=int)
    if kind == 1:
        return rng.integers(0, 2, size=height)
    if kind == 2:
        return (np.arange(height) >= int(rng.integers(1, height))).astype(int)
    return np.minimum(2, np.arange(height) * 3 // height)


def _new_piece(body):
    return dict(body=set(body), pins=set())


def _joint(pieces, a, b, cells):
    """Use cells owned by a as shared registration cells on the a/b seam."""
    for q in cells:
        assert q in pieces[a]['body'] and q not in pieces[b]['body']
        pieces[a]['body'].remove(q)
        pieces[a]['pins'].add(q)
        pieces[b]['pins'].add(q)


def _layout(rng, style, right_count, split_right):
    """Invent a mosaic directly; never chain either parent puzzle generator.

    The left panels carry two counterexamples. Extra right panels and an
    optional subdivided panel introduce two-sided branching and propagation.
    """
    h = int(rng.integers(4, 6)) if style != 'combined' else 4
    count = 4 if style == 'combined' else (3 if style == 'varied' else 2)
    heights = [h] * count
    if style == 'varied':
        heights[-1] = int(rng.integers(3, 6))
    H = sum(heights)
    if style in ('closure', 'combined'):
        first = int(rng.integers(2, 5))
        widths = [first, first + int(rng.integers(1, 3))]
        rng.shuffle(widths)
    else:
        widths = [int(rng.integers(2, 6))] * 2
    if count > 2:
        width = int(rng.integers(2, 6))
        widths += [width] * (count - 2)
    profile_a, profile_b = _tooth(rng, h), _tooth(rng, h)
    profiles = [profile_a.copy(), profile_a.copy()]
    profiles += [profile_b.copy() for _ in range(count - 2)]
    if style == 'varied':
        profiles[-1] = _tooth(rng, heights[-1])
    left = np.concatenate([w + p for w, p in zip(widths, profiles)])
    right = np.zeros(H, dtype=int)
    right_ranges = []
    if right_count:
        cut = int(rng.integers(3, H - 2)) if right_count == 2 else H
        bounds = [0, cut, H] if right_count == 2 else [0, H]
        for a, b in zip(bounds, bounds[1:]):
            w = int(rng.integers(2 if split_right else 1, 5))
            p = _tooth(rng, b - a)
            right[a:b] = w + p
            right_ranges.append((a, b))
    W = int(left.max() + right.max() + int(rng.integers(3, 6)))
    if W > 18:
        raise ValueError('layout too wide')
    core = {(r, c) for r in range(H) for c in range(int(left[r]), W - int(right[r]))}
    pieces = [_new_piece(core)]
    start = 0
    # Equal-cardinality keys; repeated whole signatures deliberately occur.
    repeat_rows = (0, 2) if style == 'combined' else tuple(sorted(
        int(x) for x in rng.choice(h, 2, replace=False)))
    row_keys = [repeat_rows, repeat_rows]
    if style == 'keys':
        row_keys = [(0, 1), (0, h - 1)]
    if count > 2:
        row_keys += [(0, 1), (0, h - 1)][:count - 2]
    for height, key in zip(heights, row_keys):
        body = {(r, c) for r in range(start, start + height)
                for c in range(int(left[r]))}
        pieces.append(_new_piece(body))
        pins = {(start + k, int(left[start + k])) for k in key}
        _joint(pieces, 0, len(pieces) - 1, pins)
        start += height
    for ri, (a, b) in enumerate(right_ranges):
        body = {(r, c) for r in range(a, b)
                for c in range(W - int(right[r]), W)}
        if split_right and ri == 0:
            # A tooth-edged inner tile plus an outer bar, with a separate key.
            cut = W - 1
            inside = {q for q in body if q[1] < cut}
            outside = body - inside
            inner_index = len(pieces)
            pieces.extend((_new_piece(inside), _new_piece(outside)))
            key_rows = sorted(int(x) for x in rng.choice(np.arange(a, b), 2, replace=False))
            _joint(pieces, inner_index, inner_index + 1,
                   {(r, cut - 1) for r in key_rows})
            child = inner_index
        else:
            child = len(pieces)
            pieces.append(_new_piece(body))
        key_rows = sorted(int(x) for x in rng.choice(np.arange(a, b), 2, replace=False))
        _joint(pieces, 0, child, {(r, W - int(right[r]) - 1) for r in key_rows})
    for p in pieces:
        if len(_components(p['body'])) != 1 or len(_components(p['body'] | p['pins'])) != 1:
            raise ValueError('joint cut disconnected a piece')
    # An internal rivet replaces a core body cell, without changing its mask.
    candidates = sorted(q for q in pieces[0]['body'] if all(
        (q[0] + dr, q[1] + dc) in pieces[0]['body'] for dr, dc in N4))
    if not candidates:
        raise ValueError('no interior rivet site')
    rivet = _choose(rng, candidates)
    return pieces, rivet, H, W


def _dilate(mask):
    result = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1, c0, c1 = max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
            result[r0 + dr:r1 + dr, c0 + dc:c1 + dc] |= mask[r0:r1, c0:c1]
    return result


def _scatter(rng, out, pieces, marker):
    height, width = out.shape
    for _ in range(30):
        inp = np.zeros_like(out)
        for q, color in pieces[0]['colors'].items():
            inp[q] = color
        for q in pieces[0]['pins']:
            inp[q] = marker
        forbidden = _dilate(inp != 0)
        order = list(range(1, len(pieces)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(pieces[i]['body'] | pieces[i]['pins']))
        for i in order:
            piece = pieces[i]
            cells = np.array(sorted(piece['body'] | piece['pins']), dtype=int)
            origin = cells.min(axis=0)
            local = cells - origin
            h, w = local.max(axis=0) + 1
            valid = np.ones((height - h + 1, width - w + 1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            if all(0 <= origin[k] < valid.shape[k] for k in (0, 1)):
                valid[tuple(origin)] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            pos = choices[int(rng.integers(len(choices)))]
            d = pos - origin
            for (r, c), color in piece['colors'].items():
                inp[r + d[0], c + d[1]] = color
            for r, c in piece['pins']:
                inp[r + d[0], c + d[1]] = marker
            forbidden = _dilate(inp != 0)
        else:
            return inp
    raise ValueError('not enough scatter clearance')


def _body_swap_witness(info, positions):
    """A silhouette-only packing can swap equal bodies of different hues.

    It preserves the payload union but discards the moving connector cells.
    Thus it explicitly tests removal of B, rather than a second full solver.
    """
    pieces, anchor = info['pieces'], info['anchor']
    for i, a in enumerate(pieces):
        if i == anchor:
            continue
        for j in range(i):
            b = pieces[j]
            if j == anchor or a['color'] == b['color']:
                continue
            if _normalized(a['body']) != _normalized(b['body']):
                continue
            ap, bp = _shift(a['body'], positions[i]), _shift(b['body'], positions[j])
            ta, la, _, _ = _box(ap)
            tb, lb, _, _ = _box(bp)
            da = (tb - ta + positions[i][0], lb - la + positions[i][1])
            db = (ta - tb + positions[j][0], la - lb + positions[j][1])
            if (_shift(a['body'], da) == bp and _shift(b['body'], db) == ap
                    and (_shift(a['pins'], da) != _shift(b['pins'], positions[j])
                         or _shift(b['pins'], db) != _shift(a['pins'], positions[i]))):
                return (i, j, da, db)
    return None


def _make_pair(rng, style, right_count, split_right, rotation, reflected, palette):
    pieces, rivet, H, W = _layout(rng, style, right_count, split_right)
    marker, anchor_color, available = palette
    colors = [anchor_color] + [int(_choose(rng, available)) for _ in pieces[1:]]
    # The B witness changes payload colors, not just decorative marker cells.
    if style == 'keys':
        colors[2] = int(_choose(rng, [c for c in available if c != colors[1]]))
    if style == 'combined':
        colors[4] = int(_choose(rng, [c for c in available if c != colors[3]]))
    margin = 3 if style in ('closure', 'combined') else 0
    height = int(rng.integers(max(20, H + 3), 31))
    width = int(rng.integers(max(20, W + margin + 2), 31))
    top = int(rng.integers(height - H + 1))
    left = int(rng.integers(margin, width - W + 1))
    offset = (top, left)
    out = np.zeros((height, width), dtype=np.int8)
    for i, p in enumerate(pieces):
        p['body'] = set(_shift(p['body'], offset))
        p['pins'] = set(_shift(p['pins'], offset))
        p['colors'] = {q: colors[i] for q in p['body']}
        for q in p['body']:
            assert out[q] == 0
            out[q] = colors[i]
    q = (rivet[0] + top, rivet[1] + left)
    out[q] = 9
    pieces[0]['colors'][q] = 9
    for p in pieces:
        for q in p['pins']:
            out[q] = marker
    inp = _scatter(rng, out, pieces, marker)
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    if reflected:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    answers, info = _assemblies(inp)
    if len(answers) != 1 or not np.array_equal(answers[0][0], out):
        raise ValueError('construction has no unique rectangular reconstruction')
    if style in ('closure', 'combined'):
        free, _ = _assemblies(inp, rectangular=False)
        if not any(not np.array_equal(answer, out) for answer, _ in free):
            raise ValueError('closure counterexample was lost')
    if style in ('keys', 'combined') and _body_swap_witness(info, answers[0][1]) is None:
        raise ValueError('key counterexample was lost')
    return dict(input=inp, output=out)


def _counts(num_train, num_test):
    for v, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not lo <= int(v) <= hi):
            raise ValueError('illegal example count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _palette(rng):
    marker, anchor = (int(x) for x in rng.choice(np.arange(1, 9), 2, replace=False))
    colors = tuple(int(x) for x in rng.permutation([c for c in range(1, 9)
                                                  if c not in (marker, anchor)]))
    return marker, anchor, colors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.integers(2))
    palette = _palette(rng)
    base_rotation = int(rng.integers(4))
    orientation_transfer = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            style = ('closure' if i == 0 else 'keys' if i == 1 else
                     _choose(rng, ['closure', 'keys', 'varied'])) if split == 'train' else 'combined'
            # At least one multistep docking example exists in training.
            if split == 'train' and i == 0:
                right_count, split_right = 1, True
            elif split == 'train' and i == 1:
                right_count, split_right = int(rng.integers(0, 3)), False
            else:
                right_count = int(rng.integers(0, 3))
                split_right = bool(right_count and rng.integers(2))
            rotation = base_rotation
            if orientation_transfer and split == 'test':
                rotation = (base_rotation + 1) % 4
            elif not orientation_transfer:
                rotation = (base_rotation + int(rng.integers(4))) % 4
            reflected = bool(rng.integers(2))
            pair_palette = palette if stable_palette else _palette(rng)
            for attempt in range(240):
                try:
                    pair = _make_pair(rng, style, right_count, split_right,
                                      rotation, reflected, pair_palette)
                except (ValueError, AssertionError):
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded mosaic construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    answers, info = _assemblies(inp)
    assert len(answers) == 1 and np.array_equal(answers[0][0], out)
    positions, pieces = answers[0][1], info['pieces']
    assert set(positions) == set(range(len(pieces)))
    assert positions[info['anchor']] == (0, 0)
    body_union, pin_counts = set(), Counter()
    redrawn = np.zeros_like(inp)
    for i, p in enumerate(pieces):
        d = positions[i]
        bp, bm = _shift(p['body'], d), _shift(p['pins'], d)
        assert not bp & body_union
        assert i == info['anchor'] or d != (0, 0)
        body_union.update(bp)
        pin_counts.update(bm)
        for (r, c), color in p['colors'].items():
            redrawn[r + d[0], c + d[1]] = color
    assert set(pin_counts.values()) == {2}
    assert not body_union & set(pin_counts)
    for q in pin_counts:
        redrawn[q] = info['marker']
    assert np.array_equal(redrawn, out)
    occupied = body_union | set(pin_counts)
    t, l, b, r = _box(occupied)
    assert len(occupied) == (b - t + 1) * (r - l + 1)
    assert np.array_equal(inp == 9, out == 9)
    for c in range(1, 10):
        if c == info['marker']:
            assert np.count_nonzero(inp == c) == 2 * np.count_nonzero(out == c)
        else:
            assert np.count_nonzero(inp == c) == np.count_nonzero(out == c)
    return info, positions


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, witnesses = set(), {'closure': False, 'keys': False}
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            info, positions = _validate_pair(pair)
            inp = _grid(pair['input'])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            if split == 'train':
                if _body_swap_witness(info, positions) is not None:
                    witnesses['keys'] = True
                if i == 0:
                    free, _ = _assemblies(inp, rectangular=False)
                    witnesses['closure'] = any(not np.array_equal(a, pair['output']) for a, _ in free)
            else:
                assert _body_swap_witness(info, positions) is not None
                free, _ = _assemblies(inp, rectangular=False)
                assert any(not np.array_equal(a, pair['output']) for a, _ in free)
    assert all(witnesses.values()), 'training must expose both inherited constraints'


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def _features(pair):
    info, positions = _validate_pair(pair)
    pieces, anchor = info['pieces'], info['anchor']
    pins = [_shift(p['pins'], positions[i]) for i, p in enumerate(pieces)]
    joints = [pins[i] & pins[j] for i in range(len(pieces)) for j in range(i)
              if pins[i] & pins[j]]
    degrees = [sum(bool(pins[i] & pins[j]) for j in range(len(pieces)) if i != j)
               for i in range(len(pieces))]
    shape = _box(map(tuple, np.argwhere(np.asarray(pair['output']) != 0).tolist()))
    bodies = [_normalized(p['body']) for p in pieces]
    return dict(canvas=tuple(np.asarray(pair['input']).shape), pieces=len(pieces),
                hull=(shape[2] - shape[0] + 1, shape[3] - shape[1] + 1),
                marker=info['marker'], core_color=pieces[anchor]['color'],
                masks=bodies, keys=[_normalized(j) for j in joints],
                degrees=degrees, repeated_colors=len({p['color'] for p in pieces}) < len(pieces),
                stepped_core=any(len({c for r, c in p['body'] if r == row}) !=
                                 len({c for r, c in p['body'] if r == other})
                                 for p in [pieces[anchor]]
                                 for row in {r for r, c in p['body']}
                                 for other in {r for r, c in p['body']}),
                nodes=info['nodes'])


def self_test():
    """Bounded semantic, API, mutation, determinism and 42-seed diversity checks."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        records.extend(_features(p) for p in puzzle['train'] + puzzle['test'])
        signatures.add(tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes())
                             for p in puzzle['train'] + puzzle['test']))
    assert len(signatures) == 42
    assert len({r['canvas'] for r in records}) >= 30
    assert len({r['hull'] for r in records}) >= 35
    assert len({mask for r in records for mask in r['masks']}) >= 100
    assert len({key for r in records for key in r['keys']}) >= 8
    assert len({r['marker'] for r in records}) >= 6
    assert len({r['core_color'] for r in records}) >= 6
    assert {r['pieces'] for r in records} >= {3, 4, 5, 6, 7}
    assert any(max(r['degrees']) >= 4 for r in records)
    assert any(r['repeated_colors'] for r in records)
    assert any(r['stepped_core'] for r in records)
    assert any(r['hull'][0] > r['hull'][1] for r in records)
    assert any(r['hull'][0] < r['hull'][1] for r in records)
    # Equal silhouettes AND equal keys leave two colored rectangular answers.
    # A tie must be rejected rather than resolved by input order or hue.
    ambiguous = np.zeros((20, 20), dtype=np.int8)
    ambiguous[6:14, 10:13] = 1
    ambiguous[9, 11] = 9
    ambiguous[[6, 8, 10, 12], 10] = 8
    for top, color in ((0, 2), (15, 3)):
        ambiguous[top:top + 4, 1:4] = color
        ambiguous[[top, top + 2], 4] = 8
    assert len(_assemblies(ambiguous)[0]) == 2
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('colored rectangular ambiguity was accepted')
    original = generate_puzzle(19, 2, 1)
    assert _equal(original, generate_puzzle(19, 2, 1))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(314159, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
    # These mutations attack two rule invariants, rather than mirroring code.
    for mutation in ('erase_joint', 'move_rivet'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in original[s]]
               for s in original}
        pair = bad['test'][0]
        _, marker, _, _ = _analyze(pair['input'])
        if mutation == 'erase_joint':
            q = tuple(np.argwhere(pair['output'] == marker)[0])
            pair['output'][q] = 0
        else:
            q = tuple(np.argwhere(pair['output'] == 9)[0])
            pair['output'][q] = 0
            pair['output'][0, 0] = 9
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('a corrupted output was accepted')
    return dict(seeds=42, pairs=len(records), distinct_canvases=len({r['canvas'] for r in records}),
                distinct_hulls=len({r['hull'] for r in records}),
                distinct_body_masks=len({m for r in records for m in r['masks']}),
                distinct_joint_keys=len({k for r in records for k in r['keys']}),
                piece_counts=sorted({r['pieces'] for r in records}),
                max_search_nodes=max(r['nodes'] for r in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
