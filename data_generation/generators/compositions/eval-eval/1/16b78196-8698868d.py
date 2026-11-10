"""Broken seals: exact interlocks whose assembled ink cavities choose a socket.

Only NumPy and the standard library are used. No files, seeds, or generation
records enter solve(). The search counts distinct colored answers, not orders
of indistinguishable fragments. See task_spec.md for the complete convention.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return grid.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _cavities(ink, diagonal=False):
    """The surrounding one-cell exterior distinguishes holes from open bays."""
    air = ~np.pad(np.asarray(ink, dtype=bool), 1)
    return [cells - 1 for cells in _components(air, diagonal)
            if not np.any((cells[:, 0] == 0) | (cells[:, 1] == 0)
                          | (cells[:, 0] == air.shape[0] - 1)
                          | (cells[:, 1] == air.shape[1] - 1))]


def _runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), 1)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(map(int, starts), map(int, ends)))


def _description(cells, values, side):
    cross = cells[:, 1] - cells[:, 1].min()
    outward = side * cells[:, 0]
    lo, hi = [], []
    for c in range(int(cross.max()) + 1):
        rows = np.sort(outward[cross == c])
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("fragment envelope must be column convex along the band")
        lo.append(int(rows[0]))
        hi.append(int(rows[-1]) + 1)
    origin = min(lo)
    return {"lo": tuple(v - origin for v in lo),
            "hi": tuple(v - origin for v in hi),
            "local": np.column_stack((outward - origin, cross)),
            "values": values}


def _interpret(grid, background, axis):
    """Parse a possible background/orientation using visible geometry only."""
    canonical = grid if axis == 0 else grid.T
    height, width = canonical.shape
    components = _components(canonical != background)
    spanning = [i for i, cells in enumerate(components)
                if len(np.unique(cells[:, 1])) == width]
    if len(spanning) != 1:
        raise ValueError("expected one band spanning the cross axis")
    band_id = spanning[0]
    cells = components[band_id]
    band_mask = np.zeros_like(canonical, dtype=bool)
    band_mask[cells[:, 0], cells[:, 1]] = True
    full = np.flatnonzero(np.all(band_mask, axis=1))
    if len(full) < 2:
        raise ValueError("missing full band edges")
    top, bottom = int(full[0]), int(full[-1])
    if not (0 < top < bottom < height - 1 and 7 <= bottom - top + 1 <= 9):
        raise ValueError("invalid band core")
    # The middle is a plain, solid spine; clue zones lie nearer the two faces.
    middle = canonical[top + 3:bottom - 2]
    if not middle.size or len(np.unique(middle)) != 1:
        raise ValueError("missing solid spine")
    substrate = int(middle.flat[0])
    colors = set(map(int, np.unique(canonical)))
    if len(colors) != 3 or substrate == background:
        raise ValueError("need background, substrate and ink")
    ink = next(iter(colors - {background, substrate}))
    sockets = []
    allowed_dots = np.zeros_like(canonical, dtype=bool)
    expected_band = np.full_like(canonical, background)
    expected_band[top:bottom + 1] = substrate
    for side, edge in ((-1, top), (1, bottom)):
        for start, end in _runs(canonical[edge] == ink):
            w = end - start
            if not 5 <= w <= 9:
                raise ValueError("socket stripe width must be 5..9")
            zone_rows = (top + 1, top + 3) if side == -1 else (bottom - 2, bottom)
            a, b = zone_rows
            zone = canonical[a:b, start:end]
            if not set(map(int, np.unique(zone))) <= {background, substrate}:
                raise ValueError("ink inside a clue zone")
            dots = zone == background
            if (np.any(dots[:, [0, -1]]) or np.any(dots[:-1] & dots[1:])
                    or np.any(dots[:, :-1] & dots[:, 1:])):
                raise ValueError("socket dots must be isolated and inside stripe ends")
            k = int(dots.sum())
            if not 0 <= k <= 4:
                raise ValueError("socket dot count outside 0..4")
            allowed_dots[a:b, start:end] |= dots
            expected_band[edge, start:end] = ink
            root = []
            for c in range(start, end):
                occupied = np.flatnonzero(band_mask[:, c])
                depth = top - int(occupied[0]) if side == -1 else int(occupied[-1]) - bottom
                if not 0 <= depth <= 3:
                    raise ValueError("root too deep")
                root.append(depth)
                if depth:
                    rr = top - np.arange(1, depth + 1) if side == -1 else bottom + np.arange(1, depth + 1)
                    expected_band[rr, c] = substrate
            if min(root) != 0 or max(root) == 0:
                raise ValueError("need a jagged root with a zero reference")
            sockets.append({"side": side, "start": start, "width": w,
                            "count": k, "root": tuple(root),
                            "anchor": top - 1 if side == -1 else bottom + 1})
    if not 1 <= len(sockets) <= 4 or len({s['count'] for s in sockets}) != len(sockets):
        raise ValueError("socket counts must be distinct")
    # Stripes on the same face have a blank band cell between them by _runs.
    expected_band[allowed_dots] = background
    actual_band = np.full_like(canonical, background)
    actual_band[band_mask] = canonical[band_mask]
    if not np.array_equal(expected_band, actual_band):
        raise ValueError("unmarked projection, clue, or band detail")
    pieces = []
    for i, source in enumerate(components):
        if i == band_id:
            continue
        values = canonical[source[:, 0], source[:, 1]].copy()
        if not set(map(int, values)) <= {substrate, ink}:
            raise ValueError("unexpected fragment color")
        descriptions = {s: _description(source, values, s) for s in (-1, 1)}
        pieces.append({"source": source, "values": values, "descriptions": descriptions})
    if not 1 <= len(pieces) <= 12:
        raise ValueError("invalid fragment count")
    return {"grid": canonical, "background": background, "substrate": substrate,
            "ink": ink, "axis": axis, "base": actual_band,
            "band_mask": band_mask, "sockets": sockets, "pieces": pieces,
            "top": top, "bottom": bottom}


def _scene(grid):
    interpretations = []
    for bg in map(int, np.unique(grid)):
        for axis in (0, 1):
            try:
                interpretations.append(_interpret(grid, bg, axis))
            except (ValueError, IndexError, StopIteration):
                pass
    if len(interpretations) != 1:
        raise ValueError("input must have exactly one visible band interpretation")
    return interpretations[0]


def _seal(placements, scene, socket):
    """Inspect the emergent pattern, excluding the fixed socket's marker stripe."""
    side, anchor = socket['side'], socket['anchor']
    all_cells = np.concatenate([cells for _, cells in placements])
    u = side * (all_cells[:, 0] - anchor)
    if np.any(u < 0):
        return None
    patch = np.zeros((int(u.max()) + 1, socket['width']), dtype=bool)
    for i, cells in placements:
        local_u = side * (cells[:, 0] - anchor)
        local_c = cells[:, 1] - socket['start']
        patch[local_u, local_c] = scene['pieces'][i]['values'] == scene['ink']
    points = np.argwhere(patch)
    if not len(points):
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    body = patch[r0:r1, c0:c1]
    if (c0 != 0 or c1 != socket['width'] or r0 <= max(socket['root'])
            or r1 != len(patch) or min(body.shape) < 4
            or not np.all(body[0]) or not np.all(body[-1])
            or not np.all(body[:, 0]) or not np.all(body[:, -1])):
        return None
    holes = _cavities(body)
    return {"count": len(holes), "sizes": tuple(sorted(map(len, holes))),
            "eight_count": len(_cavities(body, True)), "ink_mask": body,
            "collar": int(r0), "depth": len(patch)}


def _assemblies(scene, use_counts=True):
    """Full-width complementary chains, then global all-fragment exact cover."""
    height, width = scene['grid'].shape
    pieces = scene['pieces']
    options = []
    for socket in scene['sockets']:
        candidates = []
        side = socket['side']
        descriptions = [p['descriptions'][side] for p in pieces]
        lookup = {}
        for i, d in enumerate(descriptions):
            if len(d['lo']) == socket['width']:
                lookup.setdefault(d['lo'], []).append(i)

        def extend(profile, used, placements):
            if used and len(set(profile)) == 1:
                seal = _seal(placements, scene, socket)
                if seal is not None and (not use_counts or seal['count'] == socket['count']):
                    candidates.append({"bits": used, "placements": placements,
                                       "seal": seal, "socket": socket})
            offset = min(profile)
            key = tuple(v - offset for v in profile)
            for i in lookup.get(key, ()):
                if used & (1 << i):
                    continue
                d = descriptions[i]
                outward = d['local'][:, 0] + offset
                rows = socket['anchor'] + side * outward
                cols = d['local'][:, 1] + socket['start']
                if (np.any(rows < 0) or np.any(rows >= height)
                        or np.any(cols < 0) or np.any(cols >= width)
                        or np.any(scene['base'][rows, cols] != scene['background'])):
                    continue
                cells = np.column_stack((rows, cols))
                if np.array_equal(cells, pieces[i]['source']):
                    continue
                # Column profiles imply disjoint occupied masks and no gaps.
                extend(tuple(v + offset for v in d['hi']), used | (1 << i),
                       placements + [(i, cells)])

        extend(socket['root'], 0, [])
        options.append(candidates)
    full = (1 << len(pieces)) - 1
    answers = {}

    def cover(pending, used, chosen):
        if len(answers) >= 2:
            return
        if not pending:
            if used != full:
                return
            answer = scene['base'].copy()
            for candidate in chosen:
                for i, cells in candidate['placements']:
                    if np.any(answer[cells[:, 0], cells[:, 1]] != scene['background']):
                        return
                    answer[cells[:, 0], cells[:, 1]] = pieces[i]['values']
            answers.setdefault(answer.tobytes(), (answer, chosen))
            return
        j = min(pending, key=lambda k: sum(not c['bits'] & used for c in options[k]))
        for candidate in options[j]:
            if not candidate['bits'] & used:
                cover([k for k in pending if k != j], used | candidate['bits'], chosen + [candidate])

    cover(list(range(len(options))), 0, [])
    return list(answers.values())


def _decode(grid):
    scene = _scene(grid)
    answers = _assemblies(scene)
    if len(answers) != 1:
        raise ValueError("need exactly one distinct colored assembly")
    answer, tabs = answers[0]
    scene['tabs'] = tabs
    return (answer if scene['axis'] == 0 else answer.T).copy(), scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the uniquely count-keyed seals, preserving the whole canvas."""
    return _decode(_grid(input_grid))[0]


def _hole_mask(rng, height, width, count, witness=False):
    holes = np.zeros((height, width), dtype=bool)
    if count == 0:
        return holes
    if count == 1 and (witness or rng.random() < .3):
        holes[1:-1, 1:-1] = True
        return holes
    if count == 2 and (witness or rng.random() < .25):
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        if rng.integers(2):
            holes = holes[::-1]
        if rng.integers(2):
            holes = holes[:, ::-1]
        return holes.copy()
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(24):
        holes[:] = False
        for index in rng.permutation(len(interior)):
            r, c = interior[int(index)]
            if any(holes[r + dr, c + dc] for dr, dc in _FOUR):
                continue
            holes[r, c] = True
            if len(_components(~holes)) != 1:
                holes[r, c] = False
            if int(holes.sum()) == count:
                break
        if int(holes.sum()) == count:
            break
    else:
        holes[:] = False
        for p in ((1, 1), (height - 2, width - 2), (1, width - 2), (height - 2, 1))[:count]:
            holes[p] = True
    target = int(rng.integers(count, (height - 2) * (width - 2) + 1))
    for index in rng.permutation(len(interior)):
        if int(holes.sum()) >= target:
            break
        r, c = interior[int(index)]
        if holes[r, c] or not any(holes[r + dr, c + dc] for dr, dc in _FOUR):
            continue
        trial = holes.copy()
        trial[r, c] = True
        if len(_components(trial)) == count and len(_components(~trial)) == 1:
            holes = trial
    assert len(_components(holes)) == count
    return holes


def _cuts(rng, root, depth, collar, body_height, number):
    cuts = [np.array(root, dtype=int)]
    width = len(root)
    for j in range(1, number):
        center = collar + int(round(body_height * j / number))
        upper = center + rng.integers(-1, 2, size=width)
        lower = cuts[-1]
        neighbor_max = np.maximum.reduce((lower, np.r_[lower[0], lower[:-1]],
                                         np.r_[lower[1:], lower[-1]]))
        upper = np.maximum(upper, neighbor_max + 1)
        if int(upper.max()) >= depth - (number - j - 1):
            raise ValueError("insufficient depth for connected interlocks")
        # A nonflat internal seam teaches actual interlocking rather than stacking.
        if len(set(map(int, upper))) == 1:
            upper[int(rng.integers(width))] += 1
        cuts.append(upper)
    cuts.append(np.full(width, depth, dtype=int))
    return cuts


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[r:r + h, c:c + w] for r in range(3) for c in range(3)])


def _scatter(rng, base, pieces, background):
    height, width = base.shape
    for _ in range(32):
        inp = base.copy()
        forbidden = _dilate(base != background)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0]) + rng.random() * 10)
        for i in order:
            cells, values = pieces[i]
            origin = cells.min(axis=0)
            local = cells - origin
            h, w = local.max(axis=0) + 1
            if h > height or w > width:
                break
            valid = np.ones((height - h + 1, width - w + 1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            if 0 <= origin[0] < valid.shape[0] and 0 <= origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            positions = np.argwhere(valid)
            if not len(positions):
                break
            target = local + positions[int(rng.integers(len(positions)))]
            inp[target[:, 0], target[:, 1]] = values
            forbidden = _dilate(inp != background)
        else:
            return inp
    raise ValueError("scatter clearance exhausted")


def _make_pair(rng, socket_count, witness, fragments, rotation):
    background, substrate, ink = map(int, rng.choice(10, 3, replace=False))
    w = int(rng.integers(6 if witness else 5, 9))
    minimum_width = socket_count * w + socket_count - 1
    width = int(rng.integers(max(20, minimum_width + 1), 31))
    slack = width - minimum_width
    # Distribute slack over margins and gaps, so no attachment stagger is fixed.
    spaces = rng.multinomial(slack, np.ones(socket_count + 1) / (socket_count + 1))
    starts = []
    cursor = int(spaces[0])
    for j in range(socket_count):
        starts.append(cursor)
        cursor += w + (1 + int(spaces[j + 1]) if j + 1 < socket_count else 0)
    root = rng.integers(0, int(rng.integers(2, 4)), size=w)
    root -= root.min()
    if not root.max():
        root[int(rng.integers(w))] = 1
    collar = int(root.max()) + 1
    counts = [1, 2] if witness else list(map(int, rng.choice(5, socket_count, replace=False)))
    rng.shuffle(counts)
    sides = [1] * socket_count
    if socket_count == 3 and rng.random() < .35:
        sides[-1] = -1
    body_heights = [int(rng.integers(6 if witness else 5, 9)) for _ in counts]
    depths = [collar + h for h in body_heights]
    core = int(rng.integers(7, 10 if len(set(sides)) == 1 else 9))
    negative = max([d for d, s in zip(depths, sides) if s == -1] or [5])
    positive = max(d for d, s in zip(depths, sides) if s == 1)
    minimum_height = core + negative + positive
    if minimum_height > 30:
        raise ValueError("tab clearance exceeds canvas")
    height = int(rng.integers(max(24, minimum_height), 31))
    top = negative + int(rng.integers(height - minimum_height + 1))
    bottom = top + core - 1
    base = np.full((height, width), background, dtype=np.int8)
    base[top:bottom + 1] = substrate
    pieces = []
    output = base.copy()
    for j, (start, side, k, mh, depth) in enumerate(zip(starts, sides, counts, body_heights, depths)):
        edge = top if side == -1 else bottom
        anchor = top - 1 if side == -1 else bottom + 1
        base[edge, start:start + w] = ink
        a = top + 1 if side == -1 else bottom - 2
        choices = [(r, c) for r in range(a, a + 2) for c in range(start + 1, start + w - 1)]
        chosen = []
        for q in rng.permutation(len(choices)):
            p = choices[int(q)]
            if all(abs(p[0] - r) + abs(p[1] - c) != 1 for r, c in chosen):
                chosen.append(p)
                if len(chosen) == k:
                    break
        if k and len(chosen) != k:
            raise ValueError("clue independent set")
        for p in chosen[:k]:
            base[p] = background
        for c, amount in enumerate(root):
            rows = anchor + side * np.arange(int(amount))
            base[rows, start + c] = substrate
        holes = _hole_mask(rng, mh, w, k, witness)
        pattern = np.full((depth, w), substrate, dtype=np.int8)
        pattern[collar:] = np.where(holes, substrate, ink)
        number = fragments if isinstance(fragments, int) else fragments[j]
        cuts = _cuts(rng, root, depth, collar, mh, number)
        for lower, upper in zip(cuts, cuts[1:]):
            local = np.array([(u, c) for c in range(w) for u in range(int(lower[c]), int(upper[c]))])
            cells = np.column_stack((anchor + side * local[:, 0], start + local[:, 1]))
            values = pattern[local[:, 0], local[:, 1]].copy()
            pieces.append((cells, values))
    output = base.copy()
    for cells, values in pieces:
        assert np.all(output[cells[:, 0], cells[:, 1]] == background)
        output[cells[:, 0], cells[:, 1]] = values
    inp = _scatter(rng, base, pieces, background)
    pair = {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(output, rotation).copy()}
    predicted, scene = _decode(pair['input'])
    if not np.array_equal(predicted, pair['output']):
        raise ValueError("construction/search disagreement")
    if witness and not _witness(scene):
        raise ValueError("missing training witness")
    return pair


def _witness(scene):
    seals = [c['seal'] for c in scene['tabs']]
    inversion = any(a['count'] < b['count'] and sum(a['sizes']) > sum(b['sizes'])
                    for a in seals for b in seals)
    connectivity = any(a['eight_count'] < a['count'] for a in seals)
    source_holes = sum(len(_cavities(_piece_ink(p, scene['ink']))) for p in scene['pieces'])
    emergent = source_holes < sum(a['count'] for a in seals)
    return inversion and connectivity and emergent and len(_assemblies(scene, False)) == 2


def _piece_ink(piece, ink):
    local = piece['source'] - piece['source'].min(axis=0)
    mask = np.zeros(tuple(local.max(axis=0) + 1), dtype=bool)
    mask[local[:, 0], local[:, 1]] = piece['values'] == ink
    return mask


def _counts(num_train, num_test):
    # Booleans, None, floats and numeric strings are not integer counts.
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) or not lo <= value <= hi):
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    nt, ne = _counts(num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    initial_rotation = int(rng.integers(4))
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            witness = split == 'train' and i == 0
            n = 2 if split == 'train' else 3
            fragments = 2 if witness else (3 if split == 'train' and i == 1
                                          else [int(rng.integers(2, 4)) for _ in range(n)])
            rotation = ((initial_rotation + i) % 4 if split == 'train' and i < 2
                        else int(rng.integers(4)))
            for _ in range(96):
                try:
                    pair = _make_pair(rng, n, witness, fragments, rotation)
                except ValueError:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded construction retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    # Return fresh arrays, with no hidden metadata in either pairs or puzzle.
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, witnesses = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, 'duplicate input'
            seen.add(key)
            predicted, scene = _decode(inp)
            assert np.array_equal(predicted, out), 'incorrect assembly'
            canonical_out = out if scene['axis'] == 0 else out.T
            fixed = scene['band_mask']
            assert np.array_equal(canonical_out[fixed], scene['grid'][fixed])
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10)), 'colors/cells were lost'
            used = set()
            for tab in scene['tabs']:
                assert tab['seal']['count'] == tab['socket']['count']
                for i, target in tab['placements']:
                    assert i not in used
                    used.add(i)
                    source = scene['pieces'][i]['source']
                    shifts = target - source
                    assert np.all(shifts == shifts[0]) and np.any(shifts[0] != 0)
                    assert np.array_equal(canonical_out[target[:, 0], target[:, 1]],
                                          scene['pieces'][i]['values'])
                # Fixed projections plus all fragments fill one solid rectangle.
                s = tab['socket']
                rows = s['anchor'] + s['side'] * np.arange(tab['seal']['depth'])
                patch = canonical_out[rows, s['start']:s['start'] + s['width']]
                assert np.all(patch != scene['background'])
            assert used == set(range(len(scene['pieces'])))
            if split == 'train':
                witnesses.append(_witness(scene))
    assert any(witnesses), 'training must demonstrate area, connectivity, seam closure and count routing'


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only hashes and aggregate structural measurements."""
    structures, shapes, palettes, roots, patterns, fingerprints = set(), set(), set(), set(), set(), set()
    counts, piece_numbers, body_shapes, sides, cavity_sizes = set(), set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                inp = pair['input']
                _, scene = _decode(inp)
                shapes.add(inp.shape)
                palettes.add((scene['background'], scene['substrate'], scene['ink']))
                roles = {scene['background']: 0, scene['substrate']: 1, scene['ink']: 2}
                normalized = bytes(roles[int(v)] for v in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
                counts.update(s['count'] for s in scene['sockets'])
                sides.add((scene['axis'], tuple(sorted(s['side'] for s in scene['sockets']))))
                for tab in scene['tabs']:
                    roots.add(tab['socket']['root'])
                    seal = tab['seal']
                    body_shapes.add(seal['ink_mask'].shape)
                    cavity_sizes.update(seal['sizes'])
                    piece_numbers.add(len(tab['placements']))
                    patterns.add(hashlib.sha256(bytes(seal['ink_mask'].shape)
                                               + seal['ink_mask'].tobytes()).hexdigest())
        if seed < 3:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(fingerprints) == 42 and len(structures) == pairs
    assert len(shapes) >= 20 and len(roots) >= 60 and len(patterns) >= 100
    assert len(palettes) >= 40 and len(body_shapes) >= 10
    assert counts == set(range(5)) and piece_numbers == {2, 3}
    assert len(sides) >= 6 and max(cavity_sizes) >= 12
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * nt + ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
            validate_generated_puzzle(puzzle)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (4, 1.5),
                   (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corruption must be detected, rather than merely checking solver execution.
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # List inputs, color permutation and whole-scene reflections must preserve
    # the visible rule; no privileged literal color or viewing direction exists.
    sample = generate_puzzle(17, 2, 1)
    lookup = np.array([7, 3, 5, 9, 1, 8, 0, 4, 6, 2], dtype=np.int8)
    for pair in sample['train'] + sample['test']:
        inp, out = lookup[pair['input']][:, ::-1], lookup[pair['output']][:, ::-1]
        assert np.array_equal(solve(inp.tolist()), out)
    return {"seeds": 42, "pairs": pairs, "normalized_structures": len(structures),
            "canvas_shapes": len(shapes), "role_palettes": len(palettes),
            "root_profiles": len(roots), "seal_patterns": len(patterns),
            "body_shapes": len(body_shapes), "max_cavity_area": max(cavity_sizes),
            "count_range": sorted(counts), "fragments_per_tab": sorted(piece_numbers)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
