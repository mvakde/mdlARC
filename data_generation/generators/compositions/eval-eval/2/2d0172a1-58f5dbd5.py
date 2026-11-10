"""Recursive cut-and-inlay plates.

Closed colored loops are scopes. Geometry supplies a compressed register;
local color equality determines which compact marks punch that plate.
No files, examples, random state, or puzzle parameters are consulted by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    p = np.pad(mask, 1)
    seen = np.zeros(p.shape, dtype=bool)
    seen[0, 0] = True
    stack = [(0, 0)]
    h, w = p.shape
    while stack:
        y, x = stack.pop()
        for r, c in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= r < h and 0 <= c < w and not p[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    return (~p & ~seen)[1:-1, 1:-1]


def _parse(grid):
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "missing background border")
    records = []
    labels = np.zeros(grid.shape, dtype=int)
    for color in sorted(int(v) for v in np.unique(grid) if v != bg):
        for cells in _components(grid == color):
            lo, hi = cells.min(0), cells.max(0)
            local = np.zeros(tuple(hi-lo+1), dtype=bool)
            local[tuple((cells-lo).T)] = True
            hole = _holes(local)
            kind = 'C' if hole.any() else 'B'
            if kind == 'C':
                _require(len(_components(hole)) == 1, "multiple holes in one loop")
            else:
                _require(local.sum() >= local.size/2, "noncompact mark")
            _require(all(d % 2 for d in local.shape), "object centers must be integral")
            full_hole = np.zeros(grid.shape, dtype=bool)
            full_hole[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = hole
            labels[tuple(cells.T)] = len(records)+1
            records.append(dict(kind=kind, color=color, cells=cells, lo=lo, hi=hi,
                                local=local, hole=full_hole, parent=None, children=[]))
    _require(bool(records), "empty drawing")
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a > 0) & (b > 0) & (a != b)), "touching objects")
    for i, child in enumerate(records):
        candidates = []
        for j, parent in enumerate(records):
            if i == j or parent['kind'] != 'C':
                continue
            inside = parent['hole'][tuple(child['cells'].T)]
            _require(inside.all() or not inside.any(), "crossing contours")
            if inside.all():
                candidates.append(j)
        if candidates:
            owner = min(candidates, key=lambda j: int(records[j]['hole'].sum()))
            child['parent'] = owner
            records[owner]['children'].append(i)
    roots = [i for i, o in enumerate(records) if o['parent'] is None]
    _require(len(roots) == 1 and records[roots[0]]['kind'] == 'C', "need one root loop")
    for obj in records:
        _require(bool(obj['children']) == (obj['kind'] == 'C'), "empty loop or enclosing mark")
    return dict(background=bg, objects=records, root=roots[0])


# A layout is (height/width, [(object identifier, row, column, rendered tile)]).
# Groups carry layout only; they never create a plate or an extra enclosure.
def _join(parts, axis, gaps):
    shape = [0, 0]
    cross = 1-axis
    shape[axis] = sum(p[0][axis] for p in parts)+sum(gaps)
    shape[cross] = max(p[0][cross] for p in parts)
    entries, cursor = [], 0
    for k, (dims, items) in enumerate(parts):
        offset = [0, 0]
        offset[axis] = cursor
        _require((shape[cross]-dims[cross]) % 2 == 0, "half-cell alignment")
        offset[cross] = (shape[cross]-dims[cross])//2
        entries.extend((obj, y+offset[0], x+offset[1], tile) for obj, y, x, tile in items)
        cursor += dims[axis]+(gaps[k] if k < len(gaps) else 0)
    return tuple(shape), entries


def _plate(layout, color, background, kinds, colors, mode='normal', ancestor_colors=()):
    dims, items = layout
    out = np.full((dims[0]+4, dims[1]+4), color, dtype=np.int8)
    # First reserve a moat around each inlaid child. Moats may share a gap.
    for obj, y, x, tile in items:
        if kinds[obj] == 'C':
            h, w = tile.shape
            out[y+1:y+h+3, x+1:x+w+3] = background
    for obj, y, x, tile in items:
        if kinds[obj] == 'C':
            h, w = tile.shape
            out[y+2:y+h+2, x+2:x+w+2] = tile
        else:
            match = colors[obj] == color
            if mode == 'all_marks':
                match = True
            elif mode == 'no_cuts':
                match = False
            elif mode == 'any_ancestor':
                match = colors[obj] in (*ancestor_colors, color)
            out[y+2, x+2] = background if match else color
    return out


def _layout_key(layout):
    return layout[0], tuple(sorted((i, y, x) for i, y, x, _ in layout[1]))


def _decode(grid, mode='normal'):
    info = _parse(grid)
    objs, bg = info['objects'], info['background']
    cache = {}
    plate_layouts = {}

    def tile(i):
        obj = objs[i]
        if obj['kind'] == 'B':
            return np.array([[obj['color']]], dtype=np.int8)
        layout = arrange(tuple(obj['children']))
        plate_layouts[i] = layout
        ancestors, p = [], obj['parent']
        while p is not None:
            ancestors.append(objs[p]['color'])
            p = objs[p]['parent']
        return _plate(layout, obj['color'], bg,
                      {j: o['kind'] for j, o in enumerate(objs)},
                      {j: o['color'] for j, o in enumerate(objs)}, mode, ancestors)

    def arrange(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            image = tile(i)
            result = image.shape, [(i, 0, 0, image)]
        else:
            candidates = []
            for axis in (0, 1):
                ordered = sorted(ids, key=lambda i: int(objs[i]['lo'][axis]))
                groups, end = [], -1
                for i in ordered:
                    if not groups or objs[i]['lo'][axis] > end+1:
                        groups.append([])
                    groups[-1].append(i)
                    end = max(end, int(objs[i]['hi'][axis]))
                if len(groups) < 2:
                    continue
                cross = 1-axis
                centers = [min(int(objs[i]['lo'][cross]) for i in g)+
                           max(int(objs[i]['hi'][cross]) for i in g) for g in groups]
                if len(set(centers)) != 1:
                    continue
                try:
                    parts = [arrange(tuple(g)) for g in groups]
                except ValueError:
                    continue
                candidates.append(_join(parts, axis, [1]*(len(parts)-1)))
            _require(bool(candidates), "no centered cardinal layout")
            result = candidates[0]
            _require(all(_layout_key(c) == _layout_key(result) for c in candidates),
                     "ambiguous whitespace decomposition")
        cache[ids] = result
        return result

    result = np.pad(tile(info['root']), 1, constant_values=bg)
    info['layouts'] = plate_layouts
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Normalize visible containment and apply each innermost plate's stencil."""
    result, _ = _decode(_grid(input_grid))
    return result


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _group(rng, items, level=0):
    if len(items) == 1:
        return items[0]
    items = list(items)
    rng.shuffle(items)
    axis = int(rng.integers(2))
    if len(items) >= 3 and level < 2 and rng.random() < .65:
        cut = int(rng.integers(1, len(items)))
        return ('G', axis, (_group(rng, items[:cut], level+1),
                            _group(rng, items[cut:], level+1)))
    return ('G', axis, tuple(items))


def _tree(rng, palette, topology):
    """Topology prescribes loop parenthood only; every region layout is sampled."""
    root_color, other = palette[:2]
    def region(index, parent_color=None):
        color = root_color if index == 0 else _pick(rng, palette)
        # The scoped witness needs adjacent colors to differ. Deeper tests may
        # repeat a grandparent's color, with independent local masks.
        if parent_color is not None:
            color = _pick(rng, [c for c in palette if c != parent_color])
        child_nodes = [region(j, color) for j in topology.get(index, [])]
        count = int(rng.integers(2, 5))
        mismatch = parent_color if parent_color is not None else other
        mark_colors = [color, mismatch]
        mark_colors += [_pick(rng, palette) for _ in range(count-2)]
        marks = [('B', c) for c in mark_colors]
        return ('C', color, _group(rng, child_nodes+marks))
    return region(0)


def _shallow(rng, palette):
    color = palette[0]
    # Four unequal-colored registers in a mixed layout expose both pitch axes.
    colors = [color, color, palette[1], _pick(rng, palette)]
    rng.shuffle(colors)
    row = ('G', 1, tuple(('B', c) for c in colors[1:]))
    return ('C', color, ('G', 0, (('B', colors[0]), row)))


def _outline(rng, h, w, py, px):
    """Four independently stepped corners and an optional inward top bay.

    All excursions lie outside the two-cell content clearance, so the path
    stays simple, 4-connected, and encloses exactly one region.
    """
    a, b = int(rng.integers(min(py, px)-1)), int(rng.integers(min(py, px)-1))
    c, d = int(rng.integers(min(py, px)-1)), int(rng.integers(min(py, px)-1))
    path = [(0, a)]
    if py >= 3 and w >= 9 and rng.random() < .65:
        u, v = w//3, 2*w//3
        depth = int(rng.integers(1, py-1))
        path += [(0, u), (depth, u), (depth, v), (0, v)]
    path += [(0, w-1-b), (b, w-1-b), (b, w-1),
             (h-1-c, w-1), (h-1-c, w-1-c), (h-1, w-1-c),
             (h-1, d), (h-1-d, d), (h-1-d, 0),
             (a, 0), (a, a), (0, a)]
    mask = np.zeros((h, w), dtype=bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _model_render(node, rng, bg, metric=False, compact=False):
    """Render the sampled tree; this path does not use the input decoder."""
    def layout(obj):
        if obj[0] == 'G':
            parts = [layout(child) for child in obj[2]]
            gaps = [int(_pick(rng, (1, 1, 3))) if metric and not compact else 1
                    for _ in parts[:-1]]
            return _join(parts, obj[1], gaps)
        image = render(obj)
        return image.shape, [(id(obj), 0, 0, image)]

    def direct(obj):
        if obj[0] == 'G':
            return [leaf for child in obj[2] for leaf in direct(child)]
        return [obj]

    def render(obj):
        if obj[0] == 'B':
            h, w = ((1, 1) if not metric or compact else
                    (_pick(rng, (1, 1, 3)), _pick(rng, (1, 1, 3))))
            image = np.full((h, w), obj[1], dtype=np.int8)
            if min(h, w) == 3 and rng.random() < .6:
                for y, x in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
                    if rng.random() < .5:
                        image[y, x] = bg
            return image
        color, content = obj[1:]
        arranged = layout(content)
        if not metric:
            children = direct(content)
            return _plate(arranged, color, bg, {id(c): c[0] for c in children},
                          {id(c): c[1] for c in children})
        py, px = ((2, 2) if compact else (int(rng.integers(2, 5)), int(rng.integers(2, 5))))
        h, w = arranged[0][0]+2*py, arranged[0][1]+2*px
        image = np.full((h, w), bg, dtype=np.int8)
        image[_outline(rng, h, w, py, px)] = color
        for _, y, x, child in arranged[1]:
            ch, cw = child.shape
            image[y+py:y+py+ch, x+px:x+px+cw] = child
        return image
    return render(node)


def _transform(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _features(info):
    objs = info['objects']
    depth, matches, mismatches, traps, mixed = 0, 0, 0, 0, False
    for i, obj in enumerate(objs):
        ancestors, p = [], obj['parent']
        while p is not None:
            ancestors.append(objs[p]['color'])
            p = objs[p]['parent']
        if obj['kind'] == 'C':
            depth = max(depth, len(ancestors)+1)
            items = info['layouts'][i][1]
            centers = [(y+(tile.shape[0]-1)//2, x+(tile.shape[1]-1)//2)
                       for _, y, x, tile in items]
            mixed |= len({y for y, _ in centers}) > 1 and len({x for _, x in centers}) > 1
        elif ancestors:
            matches += obj['color'] == ancestors[0]
            mismatches += obj['color'] != ancestors[0]
            traps += obj['color'] != ancestors[0] and obj['color'] in ancestors[1:]
    return dict(depth=depth, plates=sum(o['kind'] == 'C' for o in objs),
                marks=sum(o['kind'] == 'B' for o in objs), matches=matches,
                mismatches=mismatches, ancestor_traps=traps, mixed=bool(mixed))


def _make_pair(rng, kind, palette, bg):
    for attempt in range(120):
        if kind == 'shallow':
            node = _shallow(rng, palette)
        elif kind == 'scoped':
            node = _tree(rng, palette, {0: [1]})
        elif kind == 'deep':
            node = _tree(rng, palette, _pick(rng, ({0: [1], 1: [2]},
                         {0: [1], 1: [2]}, {0: [1, 3], 1: [2]}, {0: [1], 1: [2, 3]})))
        elif kind == 'branch':
            node = _tree(rng, palette, _pick(rng, ({0: [1, 2]}, {0: [1, 2, 3]})))
        elif kind == 'ordinary_chain':
            node = _tree(rng, palette, _pick(rng, ({}, {0: [1]})))
        else:
            node = _tree(rng, palette, _pick(rng, ({}, {0: [1]}, {0: [1, 2]})))
        # Progressive fit control affects metric details, never the rule.
        metric = _model_render(node, rng, bg, metric=True, compact=attempt >= 12)
        expected = _model_render(node, rng, bg)
        if max(*metric.shape, *expected.shape) > 28:
            continue
        code = int(rng.integers(8))
        metric, expected = _transform(metric, code), _transform(expected, code)
        pads = []
        for dim in metric.shape:
            total = int(rng.integers(2, min(7, 30-dim)+1))
            first = int(rng.integers(1, total))
            pads.append((first, total-first))
        inp = np.pad(metric, pads, constant_values=bg)
        out = np.pad(expected, 1, constant_values=bg)
        try:
            inferred, info = _decode(inp)
            _require(np.array_equal(inferred, out), "model/visible layout disagreement")
            f = _features(info)
            if kind == 'scoped':
                _require(f['ancestor_traps'] > 0, "missing local-scope witness")
                _require(f['mixed'], "scope witness needs a mixed layout")
                _require(all(not np.array_equal(out, _transform(out, j)) for j in range(1, 8)),
                         "scope witness leaves orientation ambiguous")
        except ValueError:
            continue
        return dict(input=inp, output=out)
    raise RuntimeError('bounded construction could not fit an unambiguous scene')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name+' must be an integer in the allowed range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    fixed_palette = [int(c) for c in rng.permutation(10)]
    stable = rng.random() < .2
    regime = _pick(rng, ('deep', 'branch'))
    extra_kind = 'ordinary_chain' if regime == 'branch' else 'ordinary'
    kinds = ['shallow', 'scoped']+[extra_kind]*(nt-2)
    for split, requested in (('train', kinds), ('test', [regime]*ne)):
        for kind in requested:
            for _ in range(24):
                colors = fixed_palette if stable else [int(c) for c in rng.permutation(10)]
                pair = _make_pair(rng, kind, colors[1:int(rng.integers(4, 7))], colors[0])
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, visible topology, exact negative stencils, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, test_features, orientation = set(), [], [], set(range(8))
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected, info = _decode(inp)
            assert np.array_equal(out, expected), 'wrong plate, stencil, inlay, or crop'
            bg, objs = info['background'], info['objects']
            assert all(np.all(edge == bg) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            assert np.all(out[1, 1:-1] == objs[info['root']]['color'])
            # Independently inspect each local tile rather than just counting
            # overall pixels. Its border survives; each point obeys its owner;
            # every child has a complete one-cell background moat.
            pending = [(info['root'], out[1:-1, 1:-1])]
            while pending:
                i, local = pending.pop()
                layout = info['layouts'][i]
                owner = objs[i]
                assert local.shape == (layout[0][0]+4, layout[0][1]+4)
                uncut = np.ones(local.shape, dtype=bool)
                assert all(np.all(edge == owner['color']) for edge in
                           (local[0], local[-1], local[:, 0], local[:, -1]))
                for j, y, x, tile in layout[1]:
                    if objs[j]['kind'] == 'B':
                        want = bg if objs[j]['color'] == owner['color'] else owner['color']
                        assert local[y+2, x+2] == want
                        if want == bg:
                            uncut[y+2, x+2] = False
                    else:
                        h, w = tile.shape
                        uncut[y+1:y+h+3, x+1:x+w+3] = False
                        assert np.all(local[y+1, x+1:x+w+3] == bg)
                        assert np.all(local[y+h+2, x+1:x+w+3] == bg)
                        assert np.all(local[y+1:y+h+3, x+1] == bg)
                        assert np.all(local[y+1:y+h+3, x+w+2] == bg)
                        pending.append((j, local[y+2:y+h+2, x+2:x+w+2]))
                assert np.all(local[uncut] == owner['color']), 'material changed outside a cut'
            f = _features(info)
            if split == 'train':
                train_features.append(f)
                orientation &= {j for j in range(8) if np.array_equal(out, _transform(out, j))}
                # Explicit omission checks: all stamps as holes, no subtraction,
                # and matching any ancestor each give a different drawing.
                assert not np.array_equal(out, _decode(inp, 'all_marks')[0])
                assert not np.array_equal(out, _decode(inp, 'no_cuts')[0])
                if f['ancestor_traps']:
                    assert not np.array_equal(out, _decode(inp, 'any_ancestor')[0])
            else:
                test_features.append(f)
    assert any(f['depth'] == 1 and f['mixed'] for f in train_features)
    assert any(f['depth'] >= 2 and f['mixed'] and f['ancestor_traps'] for f in train_features)
    assert orientation == {0}, 'orientation not established by training'
    assert (all(f['depth'] > max(t['depth'] for t in train_features) for f in test_features)
            or all(f['plates'] > max(t['plates'] for t in train_features) for f in test_features)), \
        'missing structural train/test shift'


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds, all counts, metamorphisms and negative mutations.

    No puzzle corpus or execution traces are written. Only a fixed small number
    of current puzzles/fixtures and bounded signature sets are retained.
    """
    signatures, input_shapes, output_shapes, structures, palettes = set(), set(), set(), set(), set()
    depths, plates, marks, blob_shapes, sizes = set(), set(), set(), set(), set()
    nonrectangular = mixed = traps = pairs = 0
    rng = np.random.default_rng(913730)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        signatures.add(signature)
        assert signature == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                prediction, info = _decode(inp)
                assert np.array_equal(prediction, out)
                f = _features(info)
                pairs += 1
                depths.add(f['depth']); plates.add(f['plates']); marks.add(f['marks'])
                mixed += f['mixed']; traps += f['ancestor_traps']
                input_shapes.add(inp.shape); output_shapes.add(out.shape)
                roles = {}
                normalized = np.array([roles.setdefault(int(c), len(roles)) for c in out.flat], dtype=np.uint8)
                structures.add((out.shape, normalized.tobytes()))
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                for obj in info['objects']:
                    if obj['kind'] == 'B':
                        blob_shapes.add(obj['local'].shape)
                    else:
                        local = obj['local']
                        nonrectangular += not all(edge.all() for edge in
                                                   (local[0], local[-1], local[:, 0], local[:, -1]))
                        sizes.add(local.shape)
        # Palette permutation preserves matching; screen orientation is retained.
        pair = puzzle['test'][0]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        for code in (1, 4):
            assert np.array_equal(solve(_transform(pair['input'], code)), _transform(pair['output'], code))
        # Same objects and topology, altered metric distances and blob areas.
        src = puzzle['train'][0]
        inp = src['input']
        bg = int(inp[0, 0])
        if max(inp.shape) <= 28:
            shifted = np.pad(inp, ((1, 1), (1, 1)), constant_values=bg)
            assert np.array_equal(solve(shifted), src['output'])
    assert len(signatures) == 42
    assert depths == {1, 2, 3} and plates == {1, 2, 3, 4}
    assert len(marks) >= 6 and len(input_shapes) >= 35 and len(output_shapes) >= 20
    assert len(structures) >= 100 and len(palettes) >= 25
    assert len(blob_shapes) >= 3 and nonrectangular >= 15 and len(sizes) >= 30
    assert mixed >= 84 and traps >= 42
    # A fixed stencil register has many metric realizations but one answer.
    # The mismatched rightmost mark still reserves a column in this register.
    fixture = ('C', 1, ('G', 1, (('B', 1), ('B', 1), ('B', 2))))
    expected = np.pad(_model_render(fixture, rng, 0), 1)
    realizations = set()
    for _ in range(8):
        metric = _model_render(fixture, rng, 0, metric=True)
        visible = np.pad(metric, 1)
        assert np.array_equal(solve(visible), expected)
        realizations.add(_key(visible))
    assert len(realizations) > 1
    small = np.pad(_model_render(fixture, rng, 0, metric=True, compact=True), 1)
    stretched = np.repeat(small, 3, axis=0)
    assert max(stretched.shape) <= 30
    assert np.array_equal(solve(stretched), expected)
    # Changing only a scoped mark's color changes exactly one normalized cell.
    nested = generate_puzzle(431, 2, 1)
    for pair in nested['train']:
        before, info = _decode(pair['input'])
        if not _features(info)['ancestor_traps']:
            continue
        objs = info['objects']
        mark = next(o for o in objs if o['kind'] == 'B' and o['parent'] is not None
                    and o['color'] != objs[o['parent']]['color']
                    and objs[o['parent']]['parent'] is not None
                    and o['color'] == objs[objs[o['parent']]['parent']]['color'])
        changed = pair['input'].copy()
        changed[tuple(mark['cells'].T)] = objs[mark['parent']]['color']
        after = solve(changed)
        assert before.shape == after.shape and np.count_nonzero(before != after) == 1
        break
    else:
        raise AssertionError('no color intervention witness')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1337, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(901, 2, 1)
    bad['test'][0]['output'][1, 1] = int(bad['test'][0]['input'][0, 0])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged plate border accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                input_shapes=len(input_shapes), output_shapes=len(output_shapes),
                normalized_outputs=len(structures), palettes=len(palettes),
                depths=sorted(depths), plate_counts=sorted(plates), mark_counts=sorted(marks),
                blob_shapes=sorted(blob_shapes), loop_sizes=len(sizes),
                nonrectangular_loops=nonrectangular, mixed_pairs=mixed, ancestor_traps=traps)
