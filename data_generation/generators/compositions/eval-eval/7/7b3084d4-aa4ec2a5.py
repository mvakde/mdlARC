"""Cavity foundry: input-derived topology constrains rigid colored assembly.

Only Python's standard library and NumPy are used. No parent modules or data
are loaded. All random construction uses the caller's local NumPy Generator.
"""
from __future__ import annotations

from collections import deque
import json

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Integer grids required")
    _require(bool(((0 <= a) & (a <= 9)).all()), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _regions(mask, neighbors=EDGE):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask, neighbors=EDGE):
    padded = np.pad(mask.astype(bool), 1)
    outside = np.zeros_like(padded)
    outside[0, 0] = True
    queue = deque([(0, 0)])
    h, w = padded.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                queue.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _clear_topology(mask):
    # Exclude checkerboards, hence diagonal-only seals/leaks or contacts.
    a, b = mask[:-1, :-1], mask[:-1, 1:]
    c, d = mask[1:, :-1], mask[1:, 1:]
    return not bool(np.any((a == d) & (b == c) & (a != b)))


def _crop(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[1 + r:1 + r + h, 1 + c:1 + c + w] for r, c in EIGHT])


def _rotations(mask):
    seen, result = set(), []
    for k in range(4):
        m = np.rot90(mask, k).copy()
        if _key(m) not in seen:
            seen.add(_key(m))
            result.append(m)
    return result


def _read(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, "Background must be uniquely most frequent")
    bg = int(values[counts.argmax()])
    singleton = [int(v) for v, n in zip(values, counts) if n == 1 and v != bg]
    _require(len(singleton) == 1, "Exactly one singleton marker required")
    marker = singleton[0]
    components = _regions(grid != bg, EIGHT)
    _require(3 <= len(components) <= 6, "One mold and two to five loose fragments required")
    _require(len(components) == len(_regions(grid != bg)), "All objects must be orthogonally connected")
    tiles, mold_array, mold_origin, carrier = [], None, None, None
    used_colors = []
    for cells in components:
        points = np.array(sorted(cells))
        lo, hi = points.min(0), points.max(0) + 1
        a = np.full(tuple(hi - lo), -1, np.int8)
        a[points[:, 0] - lo[0], points[:, 1] - lo[1]] = grid[points[:, 0], points[:, 1]]
        vals = set(map(int, a[a >= 0]))
        if marker in vals:
            _require(mold_array is None and len(vals) == 2, "Marker belongs to one monochromatic mold")
            mold_array, mold_origin = a, tuple(map(int, lo))
            carrier = next(iter(vals - {marker}))
            used_colors.append(carrier)
        else:
            _require(len(vals) == 1 and len(cells) >= 2, "Loose fragments are monochromatic and have area at least two")
            color = next(iter(vals))
            used_colors.append(color)
            mask = a >= 0
            _require(not _holes(mask).any(), "Loose fragments cannot themselves enclose holes")
            tiles.append((mask, color))
    _require(mold_array is not None and len(set(used_colors)) == len(used_colors), "Distinct object carrier colors required")
    original = mold_array >= 0
    _require(min(original.shape) >= 3 and max(original.shape) <= 28, "Mold must allow a one-cell output margin")
    # Loose fragments lie wholly outside the mold's bounding box.
    r, c = mold_origin
    view = grid[r:r + original.shape[0], c:c + original.shape[1]]
    _require(np.all(view[~original] == bg), "Mold negative space must initially be empty")
    orientations = [(k, np.rot90(mold_array, k)) for k in range(4) if np.rot90(mold_array, k)[0, 0] == marker]
    _require(len(orientations) == 1, "Marker must occupy one mold bounding-box corner")
    k, canonical = orientations[0]
    mold = canonical >= 0
    target = _holes(mold)
    _require(_clear_topology(mold) and np.array_equal(target, _holes(mold, EIGHT)), "Unambiguous original mold topology required")
    cavities = _regions(target)
    _require(1 <= len(cavities) <= 3, "Mold must contain one to three sealed cavities")
    _require(sum(int(m.sum()) for m, _ in tiles) == int(target.sum()), "Fragment area must equal sealed cavity area")
    return dict(grid=grid, background=bg, marker=marker, carrier=carrier,
                mold=mold, target=target, tiles=tiles, cavities=cavities,
                canonical_rotation=k, original_mold=original, mold_origin=mold_origin)


def _packings(target, tiles, limit=2, allow_rotation=True):
    """Enumerate distinct colored rigid covers of only the target cells.

    No pruning budget or generation state is used: this is an exact search.
    Different rotations of a symmetric monochromatic fragment are deduplicated.
    """
    h, w = target.shape
    full = sum(1 << (int(r) * w + int(c)) for r, c in np.argwhere(target))
    candidates = []
    for piece, _ in tiles:
        placements = {}
        for mask in _rotations(piece) if allow_rotation else [piece]:
            ph, pw = mask.shape
            coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
            for r in range(h - ph + 1):
                for c in range(w - pw + 1):
                    if all(target[r + pr, c + pc] for pr, pc in coords):
                        bits = sum(1 << ((r + pr) * w + c + pc) for pr, pc in coords)
                        placements[bits] = bits
        candidates.append(tuple(sorted(placements)))
    if any(not x for x in candidates):
        return []
    chosen = [0] * len(tiles)
    results, signatures = [], set()

    def search(occupied, remaining):
        if len(results) >= limit:
            return
        if not remaining:
            if occupied == full and tuple(chosen) not in signatures:
                signatures.add(tuple(chosen))
                answer = np.full((h, w), -1, np.int8)
                for i, bits in enumerate(chosen):
                    while bits:
                        bit = bits & -bits
                        answer.flat[bit.bit_length() - 1] = tiles[i][1]
                        bits ^= bit
                results.append(answer)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        options = []
        for i in remaining:
            fits = [p for p in candidates[i] if p & first and not p & occupied]
            if fits:
                options.append((len(fits), i, fits))
        for _, i, fits in sorted(options):
            rest = tuple(j for j in remaining if j != i)
            for bits in fits:
                chosen[i] = bits
                search(occupied | bits, rest)
                if len(results) >= limit:
                    return
            chosen[i] = 0

    search(0, tuple(range(len(tiles))))
    return results


def _render(facts, filling):
    body = np.pad(facts['mold'], 1)
    cavities = np.pad(facts['target'], 1)
    outline = _dilate(body) & ~body & ~cavities
    out = np.full(body.shape, facts['background'], np.int8)
    out[outline] = facts['carrier']
    out[body] = facts['marker']
    inner = out[1:-1, 1:-1]
    inner[facts['target']] = filling[facts['target']]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input only; invalid or ambiguous inputs raise AssertionError."""
    facts = _read(input_grid)
    covers = _packings(facts['target'], facts['tiles'])
    _require(len(covers) == 1, "Exactly one colored rotation-only cavity packing required")
    return _render(facts, covers[0])


def _cavity_shape(rng, large=False):
    if large:
        # One genuine scale extrapolation axis, subsequently freely rotated.
        h, w = int(rng.integers(6, 8)), int(rng.integers(4, 8))
    else:
        h, w = map(int, rng.integers(3, 6, size=2))
    mask = np.ones((h, w), bool)
    style = int(rng.integers(4))
    if style == 1:
        # One concave step, with unequal arms.
        mask[int(rng.integers(1, h)):, int(rng.integers(1, w)):] = False
    elif style == 2:
        # Orthogonally convex staircase with independently sampled row ends.
        end = w
        for r in range(h):
            end = max(1, min(w, end + int(rng.integers(-1, 2))))
            mask[r, end:] = False
    elif style == 3:
        # Bent chamber, not a rectangular frame template.
        mask[:] = False
        a, b = int(rng.integers(1, h)), int(rng.integers(1, w))
        mask[:a] = True
        mask[:, :b] = True
    return _crop(np.rot90(mask, int(rng.integers(4))))


def _partition(rng, cavity, count):
    """Grow connected labels from freely selected points; reject weak partitions."""
    if count == 1:
        return [cavity.copy()]
    cells = list(map(tuple, np.argwhere(cavity).tolist()))
    if len(cells) < 2 * count:
        return None
    for _ in range(50):
        labels = np.full(cavity.shape, -1, int)
        seeds = rng.choice(len(cells), count, replace=False)
        for i, s in enumerate(seeds):
            labels[cells[int(s)]] = i
        weights = rng.uniform(.4, 2.5, count)
        for _step in range(len(cells) - count):
            frontier = [[] for _ in range(count)]
            for r, c in cells:
                if labels[r, c] >= 0:
                    continue
                adjacent = {int(labels[rr, cc]) for dr, dc in EDGE
                            if 0 <= (rr := r + dr) < cavity.shape[0]
                            and 0 <= (cc := c + dc) < cavity.shape[1] and labels[rr, cc] >= 0}
                for i in sorted(adjacent):
                    frontier[i].append((r, c))
            active = [i for i in range(count) if frontier[i]]
            if not active:
                break
            i = min(active, key=lambda j: np.sum(labels == j) / weights[j] + float(rng.random()))
            labels[frontier[i][int(rng.integers(len(frontier[i])))]] = i
        parts = [labels == i for i in range(count)]
        if all(p.sum() >= 2 and len(_regions(p)) == 1 and not _holes(p).any() for p in parts):
            return parts
    return None


def _build_mold(rng, cavities, teaching=False):
    """Carve closed chambers and an open counterpart without changing topology."""
    if teaching:
        chamber = cavities[0]
        h, w = chamber.shape
        mold = np.ones((h + 4, 2 * w + 5), bool)
        mold[2:2 + h, 2:2 + w] &= ~chamber
        r, c = 2, w + 3
        mold[r:r + h, c:c + w] &= ~chamber
        # Connect the copied chamber to the right exterior at its first row.
        pr, pc = np.argwhere(chamber)[0]
        mold[r + pr, c + pc:] = False
        placements = [(2, 2)]
    else:
        area = sum(p.size for p in cavities)
        base = max(max(max(p.shape) for p in cavities) + 4, int(np.sqrt(area * 2.8)) + 4)
        h = min(19, max(9, base + int(rng.integers(-1, 3))))
        w = min(19, max(9, base + int(rng.integers(-1, 3))))
        mold = np.ones((h, w), bool)
        blocked = np.zeros_like(mold)
        placements = []
        for chamber in cavities:
            ph, pw = chamber.shape
            options = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)
                       if not blocked[r:r + ph, c:c + pw].any()]
            if not options:
                return None
            r, c = options[int(rng.integers(len(options)))]
            mold[r:r + ph, c:c + pw] &= ~chamber
            blocked[r - 1:r + ph + 1, c - 1:c + pw + 1] = True
            placements.append((r, c))
        original_holes = _holes(mold)
        # A pocket with a narrow door, with varying side and dimensions.
        for _ in range(60):
            k = int(rng.integers(4))
            work = np.rot90(mold, k).copy()
            hh, ww = work.shape
            ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            r = int(rng.integers(1, hh - ph))
            c = max(1, ww - pw - int(rng.integers(1, 4)))
            work[r:r + ph, c:c + pw] = False
            door = r + int(rng.integers(ph))
            work[door, c + pw:] = False
            candidate = np.rot90(work, -k).copy()
            if (candidate[0, 0] and len(_regions(candidate)) == 1 and _clear_topology(candidate)
                    and np.array_equal(_holes(candidate), original_holes)):
                mold = candidate
                break
        else:
            return None
    # Optional exterior steps/chamfers; preserve every pre-existing chamber.
    holes = _holes(mold)
    for _ in range(int(rng.integers(0, 5))):
        k = int(rng.integers(4))
        work = np.rot90(mold, k).copy()
        depth = int(rng.integers(1, 3))
        start = int(rng.integers(1, work.shape[1] - 1))
        work[:depth, start:min(work.shape[1], start + int(rng.integers(1, 4)))] = False
        candidate = np.rot90(work, -k).copy()
        if (candidate[0, 0] and _clear_topology(candidate) and len(_regions(candidate)) == 1
                and np.array_equal(_holes(candidate), holes)):
            mold = candidate
    if not _clear_topology(mold) or len(_regions(mold)) != 1:
        return None
    expected = np.zeros_like(mold)
    for chamber, (r, c) in zip(cavities, placements):
        expected[r:r + chamber.shape[0], c:c + chamber.shape[1]] |= chamber
    if not np.array_equal(_holes(mold), expected):
        return None
    # Keep the canonical bounding rectangle and upper-left anchor intact.
    if _crop(mold).shape != mold.shape or not mold[0, 0]:
        return None
    return mold, placements


def _scatter(rng, objects, background):
    """Pack tight object boxes freely, keeping every source negative space empty."""
    order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
    area = sum(p.size for p in objects)
    for attempt in range(36):
        if attempt < 18:
            expansion = float(rng.uniform(1.7, 2.8))
            ratio = float(rng.uniform(.7, 1.4))
            h = min(30, max(max(p.shape[0] for p in objects), int(np.sqrt(area * expansion * ratio))))
            w = min(30, max(max(p.shape[1] for p in objects), int(np.sqrt(area * expansion / ratio))))
        else:
            h = w = 30
        canvas = np.full((h, w), background, np.int8)
        occupied = np.zeros((h, w), bool)
        okay = True
        for i in order:
            p = objects[i]
            ph, pw = p.shape
            prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = prefix[ph:, pw:] - prefix[:-ph, pw:] - prefix[ph:, :-pw] + prefix[:-ph, :-pw]
            options = np.argwhere(sums == 0)
            if not len(options):
                okay = False
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            canvas[r:r + ph, c:c + pw][p >= 0] = p[p >= 0]
            occupied[max(0, r - 1):min(h, r + ph + 1), max(0, c - 1):min(w, c + pw + 1)] = True
        if okay:
            # Trim randomly, without cropping any foreground object.
            rr, cc = np.nonzero(canvas != background)
            top = max(0, int(rr.min()) - int(rng.integers(0, 4)))
            left = max(0, int(cc.min()) - int(rng.integers(0, 4)))
            bottom = min(h, int(rr.max()) + 1 + int(rng.integers(0, 4)))
            right = min(w, int(cc.max()) + 1 + int(rng.integers(0, 4)))
            cropped = canvas[top:bottom, left:right].copy()
            values, counts = np.unique(cropped, return_counts=True)
            if counts[values == background][0] > max(counts[values != background]):
                return cropped
            # The full packed canvas often has enough unambiguous background.
            values, counts = np.unique(canvas, return_counts=True)
            if counts[values == background][0] > max(counts[values != background]):
                return canvas
    return None


def _make_pair(rng, colors, chamber_count, large, mold_rotation, teaching=False, require_rotation=False):
    bg, marker, carrier = colors[:3]
    for _ in range(180):
        cavities = [_cavity_shape(rng, large) for _ in range(chamber_count)]
        if teaching:
            chamber = cavities[0]
            # Its open copy needs an interior cell beyond the outline. A
            # solid 3x3 subblock provides this witness before any carving.
            if not any(chamber[r:r + 3, c:c + 3].all()
                       for r in range(chamber.shape[0] - 2)
                       for c in range(chamber.shape[1] - 2)):
                continue
        counts = [2] + [1] * (chamber_count - 1)
        if chamber_count < 3 and rng.random() < .35:
            counts[int(rng.integers(chamber_count))] += 1
        elif chamber_count == 3 and rng.random() < .5:
            counts[int(rng.integers(chamber_count))] += 1
        parts = []
        for chamber, n in zip(cavities, counts):
            partition = _partition(rng, chamber, n)
            if partition is None:
                break
            parts.extend(_crop(p) for p in partition)
        else:
            built = _build_mold(rng, cavities, teaching)
            if built is None:
                continue
            mold, _ = built
            tiles = [(p, int(color)) for p, color in zip(parts, colors[3:])]
            target = _holes(mold)
            if teaching and not (~mold & ~target & ~_dilate(mold)).any():
                # The bay must retain a visible background interior after
                # outlining, not disappear entirely into its one-cell rim.
                continue
            solutions = _packings(target, tiles)
            if len(solutions) != 1:
                continue
            colored = np.full(mold.shape, -1, np.int8)
            colored[mold] = carrier
            colored[0, 0] = marker
            objects = [np.rot90(colored, mold_rotation).copy()]
            source_tiles = []
            for mask, color in tiles:
                source = np.rot90(mask, int(rng.integers(4))).copy()
                source_tiles.append((source, color))
                objects.append(np.where(source, color, -1).astype(np.int8))
            if require_rotation and _packings(target, source_tiles, allow_rotation=False):
                continue
            inp = _scatter(rng, objects, bg)
            if inp is None:
                continue
            facts = dict(mold=mold, target=target, background=bg, marker=marker, carrier=carrier)
            out = _render(facts, solutions[0])
            return {'input': inp, 'output': out}
    raise RuntimeError("Bounded construction exhausted for cavity foundry")


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer between {lo} and {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    marker = palette[0]
    variable_palette = bool(rng.integers(2))
    shift = str(rng.choice(['chambers', 'scale', 'orientation']))
    heldout = int(rng.integers(4))
    permitted = [k for k in range(4) if k != heldout]
    rng.shuffle(permitted)
    result, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', nq)):
        for i in range(n):
            if split == 'train':
                chambers = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                large = False
                rotation = permitted[i % len(permitted)] if shift == 'orientation' else int(rng.integers(4))
            else:
                chambers = 3 if shift == 'chambers' else int(rng.integers(1, 3))
                large = shift == 'scale'
                rotation = heldout if shift == 'orientation' else int(rng.integers(4))
            if variable_palette:
                other = list(map(int, rng.permutation([c for c in range(10) if c != marker])))
            else:
                other = palette[1:]
            colors = [other[0], marker, other[1]] + other[2:]
            for _ in range(12):
                pair = _make_pair(rng, colors, chambers, large, rotation,
                                  teaching=split == 'train' and i == 0,
                                  require_rotation=split == 'train' and i < 2)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not produce a distinct input')
    return result


def _features(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    f = _read(inp)
    inner = out[1:-1, 1:-1]
    per_cavity = [len(set(int(inner[r, c]) for r, c in cells)) for cells in f['cavities']]
    pieces = [m for m, _ in f['tiles']]
    exterior = ~f['mold'] & ~f['target']
    bay_far = exterior & ~_dilate(f['mold'])
    cav_shapes = []
    for cells in f['cavities']:
        m = np.zeros_like(f['target'])
        for r, c in cells:
            m[r, c] = True
        cav_shapes.append(_crop(m))
    return dict(cavities=len(cav_shapes), fragments=len(pieces),
                split_cavity=max(per_cavity) >= 2,
                needs_rotation=not bool(_packings(f['target'], f['tiles'], allow_rotation=False)),
                irregular_cavities=sum(not p.all() for p in cav_shapes),
                concave_pieces=sum(not p.all() for p in pieces),
                symmetric_pieces=sum(len(_rotations(p)) < 4 for p in pieces),
                mold_shape=f['mold'].shape, input_shape=inp.shape,
                mold_rotation=f['canonical_rotation'],
                max_cavity_axis=max(max(p.shape) for p in cav_shapes),
                open_bay_cells=int(exterior.sum()), far_bay_cells=int(bay_far.sum()),
                tile_areas=tuple(sorted(int(p.sum()) for p in pieces)),
                palette=(f['background'], f['marker'], f['carrier']),
                cavity_signatures=tuple(_key(p) for p in cav_shapes),
                piece_signatures=tuple(_key(p) for p in pieces))


def _open_copy(facts):
    """Input-only witness: a translated chamber shape fits in an open bay.

    Return the exterior destination mask. It has the same rigid covers as the
    real cavity, exposing the answer change from discarding topology.
    """
    if len(facts['cavities']) != 1:
        return None
    chamber = _crop(facts['target'])
    exterior = ~facts['mold'] & ~facts['target']
    far = exterior & ~_dilate(facts['mold'])
    h, w = exterior.shape
    ph, pw = chamber.shape
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            if np.all(exterior[r:r + ph, c:c + pw][chamber]) and np.any(far[r:r + ph, c:c + pw][chamber]):
                copy = np.zeros_like(exterior)
                copy[r:r + ph, c:c + pw] = chamber
                return copy
    return None


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Puzzle keys must be train/test")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _counts(nt, nq)
    seen, markers, records = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Pairs must have exactly input/output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Inputs must be distinct")
            seen.add(_key(inp))
            f = _read(inp)
            markers.add(f['marker'])
            _require(out.shape == tuple(x + 2 for x in f['mold'].shape), "Output must frame the canonical mold with one cell")
            covers = _packings(f['target'], f['tiles'])
            _require(len(covers) == 1, "Ambiguous or impossible fragment assembly")
            body = np.pad(f['mold'], 1)
            cavity = np.pad(f['target'], 1)
            # Separate zone checks, rather than only invoking solve().
            neighbor = np.zeros_like(body)
            for r, c in np.argwhere(body):
                for dr, dc in EIGHT:
                    rr, cc = int(r) + dr, int(c) + dc
                    if 0 <= rr < len(body) and 0 <= cc < body.shape[1]:
                        neighbor[rr, cc] = True
            outline = neighbor & ~body & ~cavity
            far = ~body & ~cavity & ~outline
            _require(np.all(out[body] == f['marker']), "Entire mold body must take marker color")
            _require(np.all(out[outline] == f['carrier']), "Exterior-only eight-neighbor outline incorrect")
            _require(np.all(out[far] == f['background']), "Far exterior or open bay wrongly filled")
            _require(np.array_equal(out[1:-1, 1:-1][f['target']], covers[0][f['target']]), "Cavity colored packing incorrect")
            for _, color in f['tiles']:
                _require(np.sum(inp == color) == np.sum(out == color), "Every fragment cell/color must be conserved")
            _require(np.sum(out == f['marker']) == np.sum(f['mold']), "Whole-body highlight area incorrect")
            _require(np.sum(out == f['carrier']) == np.sum(outline), "Old mold color must occur only in exterior outline")
            _require(np.array_equal(solve(inp), out), "Input-only solve disagrees")
            records[split].append(_features(pair))
    _require(len(markers) == 1, "Marker convention must be consistent within puzzle")
    _require(any(f['split_cavity'] for f in records['train']), "Training must demonstrate several pieces in one cavity")
    _require(any(f['needs_rotation'] for f in records['train']), "Training must rule out translation-only assembly")
    _require(any(f['far_bay_cells'] for f in records['train']), "Training must expose an open bay that remains unfilled")
    _require({1, 2} <= {f['cavities'] for f in records['train']}, "Training must establish cavity-count variation")
    teaching = _read(puzzle['train'][0]['input'])
    copy = _open_copy(teaching)
    _require(copy is not None, "First demonstration must contrast a chamber with its open copy")
    _require(bool(_packings(copy, teaching['tiles'])), "Open-copy alternative must admit the same rigid fragments")
    train_rotations = {f['mold_rotation'] for f in records['train']}
    train_scale = max(f['max_cavity_axis'] for f in records['train'])
    for f in records['test']:
        _require(f['cavities'] == 3 or f['max_cavity_axis'] > train_scale
                 or f['mold_rotation'] not in train_rotations,
                 "Test must extend chamber count, cavity scale or anchor orientation")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed geometry audit plus count, mutation and rotation checks."""
    all_f, molds, inputs, cavities, pieces, palettes, puzzles = [], set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                all_f.append(f)
                molds.add(f['mold_shape'])
                inputs.add(f['input_shape'])
                cavities.update(f['cavity_signatures'])
                pieces.update(f['piece_signatures'])
                palettes.add(f['palette'])
    _require(len(puzzles) == 42, "Seeds must produce different puzzles")
    _require(len(inputs) >= 35 and len(molds) >= 15, "Canvas and mold geometry lack variation")
    _require(len(cavities) >= 50 and len(pieces) >= 100, "Cavity and fragment silhouettes lack variation")
    _require({1, 2, 3} == {f['cavities'] for f in all_f}, "All chamber counts must occur")
    _require({2, 3, 4, 5} == {f['fragments'] for f in all_f}, "All fragment counts must occur")
    _require({0, 1, 2, 3} == {f['mold_rotation'] for f in all_f}, "Every anchor orientation must occur")
    _require(sum(f['irregular_cavities'] for f in all_f) >= 35, "Exercise irregular cavities")
    _require(sum(f['concave_pieces'] for f in all_f) >= 50, "Exercise interlocking pieces")
    _require(sum(f['symmetric_pieces'] for f in all_f) >= 10, "Exercise harmless symmetric fragments")
    _require(len({f['tile_areas'] for f in all_f}) >= 50 and len(palettes) >= 35, "Area and palette variation missing")
    reference = generate_puzzle(707, 3, 1)
    _require(_same(reference, generate_puzzle(707, 3, 1)), "Seed determinism failed")
    _require(not _same(reference, generate_puzzle(708, 3, 1)), "Different seeds repeated")
    for counts in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(800, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Rigidly rotating the entire scatter changes no canonical answer.
    pair = reference['train'][0]
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(pair['input'], k)), pair['output']), "Global rotation invariance failed")
    # Corrupt each semantic output zone and require the validator to reject it.
    f = _read(pair['input'])
    copy = _open_copy(f)
    _require(copy is not None and bool(_packings(copy, f['tiles'])), "Parent B ablation witness missing")
    # Ignoring sealing would move the same colored mosaic into this bay.
    wrong = pair['output'].copy()
    wrong_inner = wrong[1:-1, 1:-1]
    wrong_inner[f['target']] = f['background']
    alternative = _packings(copy, f['tiles'], limit=1)[0]
    wrong_inner[copy] = alternative[copy]
    _require(not np.array_equal(wrong, pair['output']), "Omitting topology must change the answer")
    _require(not _packings(f['target'], f['tiles'], allow_rotation=False), "Parent A rotation ablation witness missing")
    body, holes = np.pad(f['mold'], 1), np.pad(f['target'], 1)
    outline = _dilate(body) & ~body & ~holes
    for zone in (body, holes, outline, ~body & ~holes & ~outline):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in reference[s]] for s in reference}
        r, c = np.argwhere(zone)[0]
        old = int(bad['train'][0]['output'][r, c])
        bad['train'][0]['output'][r, c] = (old + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Semantic zone corruption accepted')
    return dict(seeds=42, pairs=len(all_f), input_shapes=len(inputs), mold_shapes=len(molds),
                cavity_shapes=len(cavities), fragment_shapes=len(pieces), palettes=len(palettes),
                cavity_counts=sorted({f['cavities'] for f in all_f}),
                fragment_counts=sorted({f['fragments'] for f in all_f}),
                irregular_cavities=sum(f['irregular_cavities'] for f in all_f),
                concave_fragments=sum(f['concave_pieces'] for f in all_f))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
