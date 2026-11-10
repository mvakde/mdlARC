"""Uncrate miniature stencils into binary mosaics, then interlock toward rails.

Only Python/NumPy are used. No evidence files, RNG state or demonstrations are
consulted by solve. Diagnostics are bounded and self_test writes no files.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _decode_canonical(g):
    """Read a top key strip and side rails. Frames may touch one another."""
    h, w = g.shape
    ground = int(g[0, -1])
    rows = [r for r in range(4, min(7, h))
            if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != ground]
    if len(rows) != 1:
        raise ValueError("no unique structural separator")
    cut = rows[0]
    n = cut - 2
    if n not in (2, 3):
        raise ValueError("tile size must be two or three")
    divider = int(g[cut, 0])
    tile = g[1:n+1, 1:n+1].copy()
    colors = np.unique(tile)
    if len(colors) != 2:
        raise ValueError("the complete tile must be binary")
    expected = np.full((cut, w), ground, np.int8)
    expected[:n+2, :n+2] = divider
    expected[1:n+1, 1:n+1] = tile
    if not np.array_equal(g[:cut], expected):
        raise ValueError("invalid visible tile frame")
    body = g[cut+1:]
    if not len(body) or w < 5:
        raise ValueError("missing yard")
    low, high = int(body[0, 0]), int(body[0, -1])
    if len({ground, divider, low, high, *map(int, colors)}) != 6:
        raise ValueError("ambiguous local palette roles")
    if not np.all(body[:, 0] == low) or not np.all(body[:, -1] == high):
        raise ValueError("open destination rails")
    yard = body[:, 1:-1].copy()
    if max(yard.shape) * n > 30:
        raise ValueError("expanded yard exceeds ARC limits")
    covered = yard == ground
    crate_regions = np.zeros(yard.shape, bool)
    objects = []
    for r in range(yard.shape[0]):
        for c in range(yard.shape[1]):
            if covered[r, c]:
                continue
            frame = int(yard[r, c])
            if frame not in (low, high) or r+2 >= len(yard) or c+2 >= yard.shape[1]:
                raise ValueError("stray pixel or incomplete crate")
            bottom, right = r+2, c+2
            while bottom < len(yard) and yard[bottom, c+1] != frame:
                bottom += 1
            while right < yard.shape[1] and yard[r+1, right] != frame:
                right += 1
            if bottom >= len(yard) or right >= yard.shape[1]:
                raise ValueError("open crate")
            patch = yard[r:bottom+1, c:right+1]
            if not (np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                    and np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)):
                raise ValueError("crate frame must be complete")
            inside = patch[1:-1, 1:-1]
            ink = np.unique(inside[inside != ground])
            if len(ink) != 1 or int(ink[0]) not in colors:
                raise ValueError("stencil must use one tile color")
            mask = inside != ground
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                raise ValueError("stencil support must be tight")
            occupied = covered[r:bottom+1, c:right+1]
            if crate_regions[r:bottom+1, c:right+1].any():
                raise ValueError("overlapping crates")
            crate_regions[r:bottom+1, c:right+1] = True
            occupied[:] = True
            objects.append(dict(r=r, c=c, mask=mask.copy(), d=int(frame == high)))
    if not objects or not np.all(covered):
        raise ValueError("missing or undecodable stencils")
    return dict(n=n, tile=tile, ground=ground, divider=divider, low=low, high=high,
                shape=yard.shape, yard=yard, objects=objects)


def _decode(g):
    candidates = []
    for transposed in (False, True):
        try:
            info = _decode_canonical(g.T if transposed else g)
            candidates.append((info, transposed))
        except (ValueError, IndexError):
            pass
    if len(candidates) != 1:
        raise ValueError("input must have one visible key-and-rail orientation")
    return candidates[0]


def _material(info, behavior="normal"):
    """Coordinates start at the stencil's miniature interior, scaled by n."""
    n, tile = info['n'], info['tile']
    a, b = map(int, np.unique(tile))
    stamp = tile if behavior == "no_exchange" else a+b-tile
    cargo = []
    for o in info['objects']:
        mask = np.ones_like(o['mask']) if behavior == "rectangle" else o['mask']
        solid = np.repeat(np.repeat(mask, n, axis=0), n, axis=1)
        patch = np.tile(stamp, mask.shape)
        direction = 0 if behavior == 'all_low' else 1-o['d'] if behavior == 'reverse_rails' else o['d']
        cargo.append(dict(r=(o['r']+1)*n, c=(o['c']+1)*n, d=direction,
                          solid=solid, patch=patch))
    return cargo


def _profiles(o):
    return {o['r']+r: (int(cs[0]), int(cs[-1]))
            for r in range(len(o['solid']))
            if len(cs := np.flatnonzero(o['solid'][r]))}


def _pack(cargo, width, behavior="normal"):
    """Rowwise precedence prevents tunneling and permits profile interlocking."""
    result = [dict(o) for o in cargo]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(cargo) if o['d'] == direction]
        indices.sort(key=lambda i: (cargo[i]['c'] if direction == 0 else
                                    -cargo[i]['c']-cargo[i]['solid'].shape[1], cargo[i]['r']))
        settled = []
        for i in indices:
            o = result[i]
            profile = ({o['r']+r: (0, o['solid'].shape[1]-1) for r in range(len(o['solid']))}
                       if behavior == 'box_packing' else _profiles(o))
            edge = 0 if direction == 0 else width-o['solid'].shape[1]
            if behavior == "no_pack":
                target = o['c']
            elif behavior == "edge_first":
                # An intentionally wrong alternative used only by witness checks.
                positions = range(edge, width-o['solid'].shape[1]+1) if direction == 0 else range(edge, -1, -1)
                target = next((x for x in positions if all(not _collision(dict(o, c=x), p) for p in settled)), None)
                if target is None:
                    raise ValueError("no placement")
            else:
                constraints = [edge]
                for p in settled:
                    pp = ({p['r']+r: (0, p['solid'].shape[1]-1) for r in range(len(p['solid']))}
                          if behavior == 'box_packing' else _profiles(p))
                    for row in profile.keys() & pp.keys():
                        if direction == 0:
                            constraints.append(p['c']+pp[row][1]+1-profile[row][0])
                        else:
                            constraints.append(p['c']+pp[row][0]-1-profile[row][1])
                target = max(constraints) if direction == 0 else min(constraints)
            if not 0 <= target <= width-o['solid'].shape[1]:
                raise ValueError("cargo exceeds yard capacity")
            if behavior not in ("edge_first", "no_pack") and ((direction == 0 and target > o['c']) or (direction == 1 and target < o['c'])):
                raise ValueError("compaction cannot push cargo away from its edge")
            o['c'] = int(target)
            settled.append(o)
    return result


def _collision(a, b):
    top, left = max(a['r'], b['r']), max(a['c'], b['c'])
    bottom = min(a['r']+len(a['solid']), b['r']+len(b['solid']))
    right = min(a['c']+a['solid'].shape[1], b['c']+b['solid'].shape[1])
    return (top < bottom and left < right and bool(np.any(
        a['solid'][top-a['r']:bottom-a['r'], left-a['c']:right-a['c']] &
        b['solid'][top-b['r']:bottom-b['r'], left-b['c']:right-b['c']])))


def _render(info, cargo):
    shape = tuple(d*info['n'] for d in info['shape'])
    output = np.full(shape, info['ground'], np.int8)
    occupied = np.zeros(shape, bool)
    for o in cargo:
        r, c = o['r'], o['c']
        h, w = o['solid'].shape
        region = np.s_[r:r+h, c:c+w]
        if r < 0 or c < 0 or r+h > shape[0] or c+w > shape[1] or np.any(occupied[region] & o['solid']):
            raise ValueError("final cargo overlap or boundary violation")
        output[region][o['solid']] = o['patch'][o['solid']]
        occupied[region] |= o['solid']
    return output


def _predict(info, behavior="normal"):
    cargo = _material(info, behavior)
    return _render(info, _pack(cargo, info['shape'][1]*info['n'], behavior))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solve; complete local geometry defines every convention."""
    info, transposed = _decode(_grid(input_grid))
    out = _predict(info)
    return out.T.copy() if transposed else out


def _simulate(info):
    """Validation by literal substitution and unit steps, without profiles."""
    n, tile = info['n'], info['tile']
    a, b = map(int, np.unique(tile))
    cargo = []
    for o in info['objects']:
        cells = {}
        for mr, mc in np.argwhere(o['mask']):
            for dr in range(n):
                for dc in range(n):
                    cells[(int(mr)*n+dr, int(mc)*n+dc)] = a+b-int(tile[dr, dc])
        cargo.append(dict(r=(o['r']+1)*n, c=(o['c']+1)*n, d=o['d'], cells=cells))
    width = info['shape'][1]*n
    for direction, delta in ((0, -1), (1, 1)):
        group = [o for o in cargo if o['d'] == direction]
        group.sort(key=lambda o: (o['c'] if direction == 0 else -o['c']-1-max(c for _, c in o['cells']), o['r']))
        settled = set()
        for o in group:
            for _ in range(width+1):
                proposed = {(o['r']+r, o['c']+c+delta) for r, c in o['cells']}
                if any(c < 0 or c >= width for _, c in proposed) or proposed & settled:
                    break
                o['c'] += delta
            else:
                raise AssertionError("unit stepping did not stop")
            settled.update((o['r']+r, o['c']+c) for r, c in o['cells'])
    out = np.full(tuple(d*n for d in info['shape']), info['ground'], np.int8)
    used = set()
    for o in cargo:
        for (r, c), value in o['cells'].items():
            point = o['r']+r, o['c']+c
            assert point not in used, "opposite groups collide"
            used.add(point)
            out[point] = value
    return out


def _mask(rng, h, w):
    mode = int(rng.integers(5))
    if mode == 0:
        mask = np.ones((h, w), bool)
    elif mode == 1:
        rr, cc = np.indices((h, w))
        mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
    elif mode == 2:
        mask = np.zeros((h, w), bool)
        for r in range(h):
            mask[r, :int(rng.integers(1, w+1))] = True
    else:
        mask = rng.random((h, w)) < rng.uniform(.3, .85)
    # Tight support, without requiring all rows or either corner to be occupied.
    mask[0, int(rng.integers(w))] = True
    mask[-1, int(rng.integers(w))] = True
    mask[int(rng.integers(h)), 0] = True
    mask[int(rng.integers(h)), -1] = True
    return mask


def _boxes_overlap(a, b):
    return (max(a['r'], b['r']) < min(a['r']+len(a['mask'])+2, b['r']+len(b['mask'])+2)
            and max(a['c'], b['c']) < min(a['c']+a['mask'].shape[1]+2, b['c']+b['mask'].shape[1]+2))


def _draw_input(info, objects):
    n, (h, w) = info['n'], info['shape']
    g = np.full((n+3+h, w+2), info['ground'], np.int8)
    g[:n+2, :n+2] = info['divider']
    g[1:n+1, 1:n+1] = info['tile']
    g[n+2] = info['divider']
    g[n+3:, 0], g[n+3:, -1] = info['low'], info['high']
    yard = g[n+3:, 1:-1]
    for o in objects:
        r, c = o['r'], o['c']
        h, w = o['mask'].shape
        yard[r:r+h+2, c:c+w+2] = info['high'] if o['d'] else info['low']
        yard[r+1:r+h+1, c+1:c+w+1] = np.where(o['mask'], int(info['tile'][0, 0]), info['ground'])
    return g


def _pair(rng, frames, divider, axis, purpose):
    for attempt in range(160):
        n = 2 if purpose == 'witness' else int(rng.choice([2, 2, 3]))
        extent = 30//n
        ordinary = [c for c in range(10) if c not in (*frames, divider)]
        ground, a, b = map(int, rng.choice(ordinary, 3, replace=False))
        bits = rng.integers(0, 2, (n, n), dtype=np.int8)
        bits[0, 0], bits[-1, -1] = 0, 1
        tile = np.where(bits, a, b).astype(np.int8)
        if purpose == 'witness':
            h, w = int(rng.integers(8, 16)), int(rng.integers(14, 16))
            top = int(rng.integers(h-6))
            start = int(rng.integers(w-13))
            masks = [np.array([[1, 0], [1, 1]], bool),
                     np.array([[0, 1], [1, 0], [1, 1]], bool),
                     np.array([[1, 0], [1, 1]], bool)]
            objects = [dict(r=top+dr, c=start+dc, mask=m, d=0)
                       for dr, dc, m in zip((0, 1, 3), (0, 5, 10), masks)]
        elif purpose == 'directions':
            h, w = int(rng.integers(7, extent+1)), int(rng.integers(8, extent+1))
            mh, mw = int(rng.integers(1, min(4, h-2)+1)), int(rng.integers(1, min(3, (w-4)//2)+1))
            m = _mask(rng, mh, mw)
            row = int(rng.integers(h-mh-1))
            objects = [dict(r=row, c=0, mask=m.copy(), d=1),
                       dict(r=row, c=w-mw-2, mask=m.copy(), d=0)]
        else:
            h, w = int(rng.integers(9, extent+1)), int(rng.integers(9, extent+1))
            count = int(rng.integers(4, 7)) if purpose == 'test' else int(rng.integers(1, 4))
            objects = []
            for _ in range(count):
                for _ in range(65):
                    mh = int(rng.integers(1, min(4, h-2)+1))
                    mw = int(rng.integers(1, min(4, w-2)+1))
                    o = dict(r=int(rng.integers(h-mh-1)), c=int(rng.integers(w-mw-1)),
                             mask=_mask(rng, mh, mw), d=int(rng.integers(2)))
                    if not any(_boxes_overlap(o, p) for p in objects):
                        objects.append(o)
                        break
                else:
                    break
            if len(objects) != count:
                continue
        info = dict(n=n, tile=tile, ground=ground, divider=divider, low=frames[0], high=frames[1], shape=(h, w))
        # Reflections of the yard vary row order and teaching-chain destination;
        # the tile remains an independent, unrotated motif.
        if bool(rng.integers(2)):
            for o in objects:
                o['r'] = h-o['r']-len(o['mask'])-2
                o['mask'] = o['mask'][::-1].copy()
        if bool(rng.integers(2)):
            for o in objects:
                o['c'] = w-o['c']-o['mask'].shape[1]-2
                o['mask'] = o['mask'][:, ::-1].copy()
                o['d'] = 1-o['d']
        inp = _draw_input(info, objects)
        try:
            visible = _decode_canonical(inp)
            out = _predict(visible)
            if purpose == 'witness':
                if any(np.array_equal(out, _predict(visible, behavior)) for behavior in ('box_packing', 'edge_first')):
                    continue
            if purpose == 'test' and np.array_equal(out, _predict(visible, 'no_pack')):
                continue
        except (ValueError, AssertionError):
            continue
        return {'input': inp.T.copy() if axis else inp, 'output': out.T.copy() if axis else out}
    raise RuntimeError(f"bounded construction exhausted for {purpose}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    low, high, divider = map(int, rng.choice(10, 3, replace=False))
    axis = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            purpose = 'test' if split == 'test' else ('witness' if i == 0 else 'directions' if i == 1 else 'free')
            for _ in range(8):
                pair = _pair(rng, (low, high), divider, axis, purpose)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, conventions, records = set(), set(), {'train': [], 'test': []}
    witnesses = {b: False for b in ('no_exchange', 'rectangle', 'box_packing', 'no_pack', 'edge_first', 'all_low', 'reverse_rails')}
    moved_classes, shared = set(), False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info, axis = _decode(inp)
            conventions.add((axis, info['low'], info['high'], info['divider']))
            canonical = out.T if axis else out
            assert np.array_equal(canonical, _predict(info)), 'incorrect output'
            assert np.array_equal(canonical, _simulate(info)), 'unit-step check failed'
            material = _material(info)
            settled = _pack(material, info['shape'][1]*info['n'])
            # All pixels in every occupied block survive, and stencil holes
            # stay background unless occupied by a different cargo silhouette.
            assert np.count_nonzero(canonical != info['ground']) == sum(int(o['solid'].sum()) for o in material)
            stamps = sum(int(o['mask'].sum()) for o in info['objects'])
            colors = np.unique(info['tile'])
            for color in colors:
                opposite = int(colors[0])+int(colors[1])-int(color)
                assert np.count_nonzero(canonical == color) == stamps*np.count_nonzero(info['tile'] == opposite)
            for a, b in zip(material, settled):
                assert a['r'] == b['r'] and np.array_equal(a['patch'], b['patch'])
                assert b['c'] % info['n'] == 0
            records[split].append(len(material))
            if split == 'train':
                moved_classes.update(a['d'] for a, b in zip(material, settled) if a['c'] != b['c'])
                shared |= any(a['d'] != b['d'] and np.array_equal(a['mask'], b['mask'])
                              for a in info['objects'] for b in info['objects'])
                for behavior in witnesses:
                    try:
                        witnesses[behavior] |= not np.array_equal(canonical, _predict(info, behavior))
                    except ValueError:
                        witnesses[behavior] = True
    assert len(conventions) == 1, 'axis or frame semantics changed within puzzle'
    assert moved_classes == {0, 1} and shared, 'training must distinguish frame from payload'
    assert all(witnesses.values()), f'missing training witnesses: {witnesses}'
    assert max(records['train']) == 3 and min(records['test']) >= 4, 'missing count shift'
    return True


def self_test():
    """42 puzzles, streamed; storage and retries are bounded, with no file writes."""
    shapes, sizes, axes, counts, masks, palettes, digests = set(), set(), set(), set(), set(), set(), set()
    interlocks = chains = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info, axis = _decode(pair['input'])
                shapes.add(pair['output'].shape)
                sizes.add(info['n'])
                axes.add(axis)
                counts.add(len(info['objects']))
                palettes.add((info['low'], info['high'], info['ground']))
                masks.update((o['mask'].shape, o['mask'].tobytes()) for o in info['objects'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                if split == 'train':
                    out = pair['output'].T if axis else pair['output']
                    interlocks += not np.array_equal(out, _predict(info, 'box_packing'))
                    chains += not np.array_equal(out, _predict(info, 'edge_first'))
        digests.add(digest.digest())
    assert len(digests) == 42 and len(shapes) >= 30 and len(masks) >= 80
    assert sizes == {2, 3} and axes == {False, True} and {1, 2, 3, 4, 5, 6} <= counts
    assert interlocks >= 42 and chains >= 42 and len(palettes) >= 42
    a, b = generate_puzzle(213), generate_puzzle(213)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    permutation = np.array([6, 2, 9, 1, 8, 3, 0, 5, 7, 4], np.int8)
    for split in a:
        for pair in a[split]:
            assert np.array_equal(solve(pair['input'].T), pair['output'].T)
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
            before = pair['input'].copy()
            solve(pair['input'])
            assert np.array_equal(before, pair['input']), 'solve mutated its argument'
    del a, b
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(720+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, distinct_puzzles=len(digests), output_shapes=len(shapes),
                stencil_patterns=len(masks), tile_sizes=sorted(sizes), orientations=len(axes),
                object_counts=sorted(counts), interlock_training_pairs=interlocks,
                stable_chain_training_pairs=chains)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
