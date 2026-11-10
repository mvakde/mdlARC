"""Pocket ballots: topology addresses a length-ranked frame; objects cast votes.

No parent code, files, seeds, or stored examples are consulted by solve().
All generation and checks are bounded and use only NumPy and the standard library.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 533761414, 752524798, 763175780, 242537144, 243228467,
              441873958, 80457273, 423071081, 408590602, 516834999, 380681002,
              843914895, 308917033, 802990175, 437081558, 752164731, 355342668,
              559364488, 170997159, 341059607, 336612747, 204765626, 39181226,
              544900754, 355432674, 418247109, 594410819, 52251812, 293860317,
              688391284, 743607639, 720272343, 307205202, 716113449, 48765217,
              236640739, 547173187, 572975705, 602793593, 798357474, 404225630)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'grid colors')
    return a


def _components(grid, background):
    seen = np.zeros(grid.shape, dtype=bool)
    parts = []
    for r, c in np.ndindex(grid.shape):
        if seen[r, c] or grid[r, c] == background:
            continue
        color, cells, stack = int(grid[r, c]), [], [(r, c)]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                        and not seen[nr, nc] and grid[nr, nc] == color):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rows, cols = zip(*cells)
        parts.append({'color': color, 'cells': tuple(cells),
                      'box': (min(rows), max(rows), min(cols), max(cols))})
    return parts


def _frame(part):
    r0, r1, c0, c1 = part['box']
    h, w = r1-r0+1, c1-c0+1
    if min(h, w) not in (5, 7) or not 9 <= max(h, w) <= 17:
        return False
    perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                 if r in (r0, r1) or c in (c0, c1)}
    return set(part['cells']) == perimeter


def _holes(part):
    """Return bounded four-connected complement regions of ONE component.

    Padding makes outside connectivity explicit; no area or color histogram is
    substituted for the number of enclosed regions.
    """
    r0, r1, c0, c1 = part['box']
    mask = np.zeros((r1-r0+3, c1-c0+3), dtype=bool)
    for r, c in part['cells']:
        mask[r-r0+1, c-c0+1] = True
    seen, holes = mask.copy(), []
    for r, c in np.ndindex(mask.shape):
        if seen[r, c]:
            continue
        stack, region, exterior = [(r, c)], [], False
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            region.append((rr+r0-1, cc+c0-1))
            exterior |= rr in (0, mask.shape[0]-1) or cc in (0, mask.shape[1]-1)
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        if not exterior:
            holes.append(tuple(region))
    return holes


def _inspect(input_grid):
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(frequencies))
    _require(np.sum(frequencies == frequencies[background]) == 1, 'ambiguous background')
    parts = _components(grid, background)
    frames = [p for p in parts if _frame(p)]
    tokens = [p for p in parts if not _frame(p)]
    _require(2 <= len(frames) <= 5 and 1 <= len(tokens) <= 8, 'frame/token count')
    verticals = {p['box'][1]-p['box'][0] > p['box'][3]-p['box'][2] for p in frames}
    _require(len(verticals) == 1, 'mixed frame axes')
    vertical = verticals.pop()
    axis = 0 if vertical else 1
    centers = [(p['box'][0]+p['box'][1])/2 if vertical else
               (p['box'][2]+p['box'][3])/2 for p in frames]
    token_centers = [sum(p[axis] for p in t['cells'])/len(t['cells']) for t in tokens]
    if min(token_centers) > max(centers):
        direction = 'down' if vertical else 'right'
    elif max(token_centers) < min(centers):
        direction = 'up' if vertical else 'left'
    else:
        raise AssertionError('tokens must identify one field-facing end')

    # An eight-neighbor background moat prevents accidental component merges
    # and objects inside another object's holes.
    labels = np.full(grid.shape, -1, dtype=int)
    for i, part in enumerate(parts):
        for r, c in part['cells']:
            labels[r, c] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, dr):grid.shape[0]+min(0, dr),
                   max(0, dc):grid.shape[1]+min(0, dc)]
        b = labels[max(0, -dr):grid.shape[0]+min(0, -dr),
                   max(0, -dc):grid.shape[1]+min(0, -dc)]
        _require(not np.any((a >= 0) & (b >= 0) & (a != b)), 'objects touch')

    families = {}
    for frame in frames:
        r0, r1, c0, c1 = frame['box']
        _require(np.all(grid[r0+1:r1, c0+1:c1] == background), 'nonempty frame')
        frame['length'] = max(r1-r0+1, c1-c0+1)
        families.setdefault(frame['color'], []).append(frame)
    _require(1 <= len(families) <= 2, 'family count')
    for family in families.values():
        _require(2 <= len(family) <= 4, 'each family needs two through four competitors')
        _require(len({min(f['box'][1]-f['box'][0]+1, f['box'][3]-f['box'][2]+1)
                      for f in family}) == 1, 'one short-side length per color family')
        _require(len({f['length'] for f in family}) == len(family), 'tied frame lengths')
        family.sort(key=lambda f: f['length'], reverse=True)
        for rank, frame in enumerate(family, 1):
            frame['rank'], frame['votes'] = rank, 0

    for token in tokens:
        r0, r1, c0, c1 = token['box']
        _require(1 <= min(r1-r0+1, c1-c0+1) <= 4
                 and max(r1-r0+1, c1-c0+1) <= 11, 'token geometry')
        holes = _holes(token)
        _require(len(holes) <= 4, 'too many token holes')
        _require(all(grid[r, c] == background for hole in holes for r, c in hole),
                 'hole contains another object')
        token['holes'] = len(holes)
        token['hole_area'] = sum(map(len, holes))
        if token['color'] in families:
            _require(1 <= len(holes) <= len(families[token['color']]), 'invalid ballot address')
            families[token['color']][len(holes)-1]['votes'] += 1
    for frame in frames:
        _require(frame['votes'] <= (frame['length']-3)//2, 'tally exceeds capacity')
    return {'background': background, 'frames': frames, 'tokens': tokens,
            'families': families, 'direction': direction}


def _draw_result(grid, detail, *, mode='intended'):
    """Alternative modes are counterfactual probes, never answer parameters."""
    out = np.full_like(grid, detail['background'])
    for frame in detail['frames']:
        for r, c in frame['cells']:
            out[r, c] = frame['color']
    for color, family in detail['families'].items():
        ballots = [t for t in detail['tokens'] if t['color'] == color]
        counts = [0]*len(family)
        for token in ballots:
            k = token['holes']
            address, weight = k-1, 1
            if mode == 'ascending':
                address = len(family)-k
            elif mode in ('spatial_low', 'spatial_high'):
                transverse = 2 if detail['direction'] in ('up', 'down') else 0
                ordered = sorted(family, key=lambda f: f['box'][transverse],
                                 reverse=mode == 'spatial_high')
                address = next(i for i, f in enumerate(family) if f is ordered[k-1])
            elif mode == 'longest':
                address = 0
            elif mode == 'third_is_shortest' and k == 3:
                address = len(family)-1
            elif mode == 'hole_votes':
                weight = k
            elif mode == 'pixel_votes':
                weight = len(token['cells'])
            elif mode == 'hole_area':
                address = token['hole_area']-1
            if 0 <= address < len(family):
                counts[address] += weight
        for frame, count in zip(family, counts):
            r0, r1, c0, c1 = frame['box']
            for i in range(1, min(count, (frame['length']-3)//2)+1):
                d = detail['direction']
                r = r1-2*i if d == 'down' else r0+2*i if d == 'up' else (r0+r1)//2
                c = c1-2*i if d == 'right' else c0+2*i if d == 'left' else (c0+c1)//2
                out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the input; preserve frames, erase tokens, tally routed votes."""
    grid = _grid(input_grid)
    return _draw_result(grid, _inspect(grid))


def _ring(widths, height=3):
    width = 1 + sum(w+1 for w in widths)
    mask = np.ones((height, width), dtype=np.int8)
    c = 1
    for w in widths:
        mask[1:-1, c:c+w] = 0
        c += w+1
    return mask


def _token(rng, holes, forced=None, small=False):
    if forced == 'wide':
        mask = _ring([5])  # area 16; two minimal one-hole rings also have area 16.
    elif forced == 'tiny':
        mask = _ring([1])
    elif holes == 0:
        mask = np.zeros((3, 3), dtype=np.int8)
        cells = {(1, 1)}
        for _ in range(int(rng.integers(0, 6))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for r, c in cells:
            mask[r, c] = 1
    else:
        widths = [1]*holes
        if not small:
            for _ in range(int(rng.integers(0, min(4, 10-2*holes)+1))):
                widths[int(rng.integers(holes))] += 1
        mask = _ring(widths, 3 if small else int(rng.integers(3, 5)))
        if not small and mask.shape[1] < 11 and rng.random() < .45:
            tail = int(rng.integers(1, min(2, 11-mask.shape[1])+1))
            mask = np.pad(mask, ((0, 0), (0, tail)))
            mask[int(rng.choice([0, mask.shape[0]-1])), -tail:] = 1
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return np.rot90(mask, int(rng.integers(4))).copy()


def _draw_frame(grid, box, color):
    r, c, h, w = box
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _make_boxes(rng, records, force_middle, fallback=False):
    order = list(rng.permutation(len(records)))
    if force_middle:
        indices = [i for i in order if records[i][0] == 0]
        first = next(i for i in indices if records[i][1] == 1)
        a, b = order.index(first), order.index(indices[1])
        order[a], order[b] = order[b], order[a]
    n = len(records)
    shorts, gaps = [5]*n, [1]*(n-1)
    extra = 30 - (6*n-1)
    if not fallback:
        for family in rng.permutation(len({record[0] for record in records})):
            positions = [j for j, index in enumerate(order) if records[index][0] == family]
            cost = 2*len(positions)
            if extra >= cost and rng.random() < .5:
                for j in positions:
                    shorts[j] = 7
                extra -= cost
        for i in range(n-1):
            add = int(rng.integers(min(2, extra)+1))
            gaps[i] += add
            extra -= add
    r = 0 if fallback else int(rng.integers(min(2, extra)+1))
    boxes = []
    aligned = fallback or rng.random() < .4
    left = 0 if fallback else int(rng.integers(2))
    for j, index in enumerate(order):
        family, rank, length, votes = records[index]
        c = left if aligned else left+int(rng.integers(2))
        boxes.append((index, (r, c, shorts[j], length)))
        r += shorts[j] + (gaps[j] if j < n-1 else 0)
    return boxes, r


def _place_tokens(rng, grid, boxes, colors, specs, *, fallback=False):
    occupied = np.zeros(grid.shape, dtype=bool)
    for index, (r, c, h, w) in boxes:
        _draw_frame(grid, (r, c, h, w), colors[index])
        occupied[max(0, r-1):min(grid.shape[0], r+h+1),
                 max(0, c-1):min(grid.shape[1], c+w+1)] = True
    edge = max(c+w for _, (r, c, h, w) in boxes)
    field_start = edge+1
    occupied[:, :field_start] = True
    ordered = sorted(range(len(specs)), key=lambda i: -specs[i][1].size)
    for step, index in enumerate(ordered):
        color, mask = specs[index]
        sh, sw = mask.shape
        rr, cc = np.nonzero(mask)
        if fallback:
            # At most eight objects, in nine slots, each at most 9 x 3.
            r, c = (step//3)*10, 19+(step%3)*4
        else:
            if sh > grid.shape[0] or sw > grid.shape[1]-field_start:
                return False
            # Block entire boxes: no token can sit inside another token's hole.
            candidates = []
            for r0 in range(grid.shape[0]-sh+1):
                for c0 in range(field_start, grid.shape[1]-sw+1):
                    if not np.any(occupied[r0:r0+sh, c0:c0+sw]):
                        candidates.append((r0, c0))
            if not candidates:
                return False
            r, c = candidates[int(rng.integers(len(candidates)))]
        if r+sh > grid.shape[0] or c+sw > grid.shape[1]:
            return False
        if np.any(occupied[r:r+sh, c:c+sw]):
            return False
        grid[r+rr, c+cc] = color
        occupied[max(0, r-1):min(grid.shape[0], r+sh+1),
                 max(0, c-1):min(grid.shape[1], c+sw+1)] = True
    return True


def _build_pair(rng, populations, counts, anchor_color=None, teaching=0, rotation=None,
                secondary_color=None):
    palette = [int(c) for c in rng.permutation(10)]
    if anchor_color is not None:
        categories = [anchor_color]+([secondary_color] if len(populations) == 2 else [])
        palette = [c for c in palette if c not in categories]
        if teaching == 2:
            palette.remove(secondary_color)
            palette.append(secondary_color)
    else:
        categories, palette = palette[:len(populations)], palette[len(populations):]
    background = palette[0]
    distractor_palette = palette[1:]
    records, colors, specs = [], [], []
    for family, (population, votes) in enumerate(zip(populations, counts)):
        lengths = sorted((int(x) for x in rng.choice(np.arange(9, 18), population,
                                                    replace=False)), reverse=True)
        for rank, (length, number) in enumerate(zip(lengths, votes), 1):
            records.append((family, rank, length, number))
            colors.append(categories[family])
            for j in range(number):
                forced = ('wide' if teaching == 1 else 'tiny') if family == 0 and rank == 1 and teaching else None
                specs.append((categories[family], rank, forced))
    # Distractors obey the same loop vocabulary, but have no frame color.
    for j in range(int(rng.integers(0 if teaching == 0 else 1, 3))):
        if len(specs) >= 8:
            break
        color = secondary_color if teaching == 2 and j == 0 else int(rng.choice(distractor_palette))
        specs.append((color, int(rng.integers(0, 5)), None))
    masks = [(color, _token(rng, holes, forced)) for color, holes, forced in specs]
    grid = None
    for _ in range(5):
        boxes, bank_height = _make_boxes(rng, records, teaching == 1)
        edge = max(c+w for _, (r, c, h, w) in boxes)
        h = min(30, max(bank_height, 12+2*len(specs))+int(rng.integers(0, 3)))
        w = min(30, edge+1+int(rng.integers(9, 13)))
        candidate = np.full((h, w), background, dtype=np.int8)
        if _place_tokens(rng, candidate, boxes, colors, masks):
            f = np.bincount(candidate.ravel(), minlength=10)
            if f[background] > max(f[c] for c in range(10) if c != background):
                grid = candidate
                break
    if grid is None:
        boxes, _ = _make_boxes(rng, records, teaching == 1, fallback=True)
        # Simplify optional shape geometry, retaining every address, vote, and
        # the exact equal-area teaching witness. Guaranteed 30 x 30 packing.
        masks = []
        for color, holes, forced in specs:
            mask = _token(rng, holes, forced, small=True)
            if mask.shape[1] > mask.shape[0]:
                mask = mask.T.copy()
            masks.append((color, mask))
        grid = np.full((30, 30), background, dtype=np.int8)
        _require(_place_tokens(rng, grid, boxes, colors, masks, fallback=True), 'fallback packing')
    # Labels are constructed from the requested census, independently of the
    # input parser and its hole counting. Validation compares these paths.
    output = np.full_like(grid, background)
    for index, box in boxes:
        r, c, h, w = box
        _draw_frame(output, box, colors[index])
        for j in range(records[index][3]):
            output[r+h//2, c+w-3-2*j] = colors[index]
    rotation = int(rng.integers(4)) if rotation is None else rotation
    reflected = bool(rng.integers(2))
    return {role: np.ascontiguousarray(np.rot90(np.flipud(g) if reflected else g, rotation))
            for role, g in (('input', grid), ('output', output))}


def _random_spec(rng, target, expansion):
    if target and expansion:
        populations = [4]
    else:
        populations = list(((2,), (3,), (2, 2), (2, 3))[int(rng.integers(4))])
    counts = [[int(x) for x in rng.integers(0, 3, p)] for p in populations]
    if target and not expansion:
        counts[0][0], counts[0][1] = 3, 0
        rng.shuffle(counts[0])
    elif target:
        counts[0][-1] = int(rng.integers(1, 3))
    if sum(map(sum, counts)) == 0:
        counts[0][int(rng.integers(populations[0]))] = 1
    while sum(map(sum, counts)) > 6:
        candidates = [(i, j) for i, row in enumerate(counts) for j, v in enumerate(row)
                      if v > (1 if target and expansion and i == 0 and j == 3 else
                              3 if target and not expansion and v == 3 else 0)]
        i, j = candidates[int(rng.integers(len(candidates)))]
        counts[i][j] -= 1
    return populations, counts


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(10))
    secondary = int(rng.choice([c for c in range(10) if c != anchor]))
    expansion = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    positions = [int(x) for x in rng.choice(num_train, 2, replace=False)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, size in (('train', num_train), ('test', num_test)):
        for index in range(size):
            teaching = positions.index(index)+1 if split == 'train' and index in positions else 0
            if teaching:
                # Both minimum-count demonstrations expose 1/2/3 addresses,
                # repeated ballots, empty destinations, and an area collision.
                if teaching == 1:
                    populations, counts = [3, 2], [[1, 2, 1], [0, 1]]
                else:
                    populations, counts = [4], [[2, 1, 1, 0]]
            else:
                populations, counts = _random_spec(rng, split == 'test', expansion)
            for _ in range(8):
                rotation = (base_rotation + 2*(teaching-1)) % 4 if teaching else None
                pair = _build_pair(rng, populations, counts, anchor if teaching else None,
                                   teaching, rotation, secondary if teaching else None)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise AssertionError('distinct-scene retry limit')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            signature = (inp.shape, inp.astype(np.int8).tobytes())
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            d = _inspect(inp)
            _require(np.array_equal(out, _draw_result(inp, d)), 'incorrect tally or erasure')
            _require(not np.array_equal(inp, out), 'identity pair')
            # A conservation check independent of tally locations: each matched
            # component must supply one and only one new nonbackground cell.
            original_frames = sum(len(f['cells']) for f in d['frames'])
            matched = sum(t['color'] in d['families'] for t in d['tokens'])
            _require(np.count_nonzero(out != d['background']) == original_frames+matched,
                     'one-object-one-vote conservation')
            details[split].append((inp, d))
    training = details['train']
    # A same-color/same-hole-address equal-area collision with different counts
    # explicitly defeats using colored area as votes.
    buckets = []
    for inp, d in training:
        for color, family in d['families'].items():
            for frame in family:
                ts = [t for t in d['tokens'] if t['color'] == color and t['holes'] == frame['rank']]
                buckets.append((color, frame['rank'], len(ts), sum(len(t['cells']) for t in ts)))
    _require(any(c == cc and k == kk and n != nn and a == aa and a > 0
                 for c, k, n, a in buckets for cc, kk, nn, aa in buckets), 'missing area collision')
    ranks = {t['holes'] for _, d in training for t in d['tokens'] if t['color'] in d['families']}
    _require({1, 2, 3} <= ranks, 'missing address progression')
    _require(any(f['votes'] >= 2 for _, d in training for f in d['frames']), 'missing vote aggregation')
    _require(any(f['votes'] == 0 for _, d in training for f in d['frames']), 'missing zero bucket')
    categories = {c for _, d in training for c in d['families']}
    discarded = {t['color'] for _, d in training for t in d['tokens']
                 if t['color'] not in d['families']}
    _require(bool(categories & discarded), 'missing visible category/distractor role change')
    for alternative in ('ascending', 'spatial_low', 'spatial_high', 'longest', 'third_is_shortest',
                        'hole_votes', 'pixel_votes', 'hole_area'):
        _require(any(not np.array_equal(_draw_result(inp, d), _draw_result(inp, d, mode=alternative))
                     for inp, d in training), f'training permits {alternative}')
    train_votes = {f['votes'] for _, d in training for f in d['frames']}
    train_ranks = {t['holes'] for _, d in training for t in d['tokens']
                   if t['color'] in d['families']}
    for inp, d in details['test']:
        _require(any(f['votes'] not in train_votes for f in d['frames'])
                 or any(t['holes'] not in train_ranks for t in d['tokens']
                        if t['color'] in d['families']),
                 'test needs a new census value or hole address')
        for token in d['tokens']:
            if token['color'] in d['families']:
                _require(token['holes'] in ranks or set(range(1, token['holes'])) <= ranks,
                         'unseen address lacks demonstrated progression')


def measure_puzzle(puzzle):
    """Small grid-derived records, with no reconstructible grid payloads."""
    records = []
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            inp = np.asarray(pair['input'])
            d = _inspect(inp)
            token_masks = []
            for t in d['tokens']:
                r0, r1, c0, c1 = t['box']
                token_masks.append(tuple(sorted((r-r0, c-c0) for r, c in t['cells'])))
            records.append({'split': split, 'index': i, 'shape': list(inp.shape),
                            'background': d['background'], 'direction': d['direction'],
                            'families': len(d['families']), 'frames': len(d['frames']),
                            'populations': sorted(len(fs) for fs in d['families'].values()),
                            'votes': [f['votes'] for f in d['frames']],
                            'ranks': [t['holes'] for t in d['tokens'] if t['color'] in d['families']],
                            'token_areas': [len(t['cells']) for t in d['tokens']],
                            'hole_areas': [t['hole_area'] for t in d['tokens']],
                            'masks': token_masks,
                            'lengths': [f['length'] for f in d['frames']],
                            'shorts': [min(f['box'][1]-f['box'][0]+1, f['box'][3]-f['box'][2]+1)
                                       for f in d['frames']],
                            'distractors': sum(t['color'] not in d['families'] for t in d['tokens'])})
    return records


def self_test():
    """42 host seeds, streamed; no files, traces, or retained puzzle archive."""
    dimensions, signatures, masks = set(), set(), set()
    backgrounds, directions, populations, votes, ranks, shorts, areas = (set() for _ in range(7))
    frame_counts, family_counts, distractor_counts, lengths = (set() for _ in range(4))
    changes = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                for role in ('input', 'output'):
                    grid = pair[role]
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
        signatures.add(digest.hexdigest())
        for m in measure_puzzle(puzzle):
            dimensions.add(tuple(m['shape']))
            masks.update(m['masks'])
            backgrounds.add(m['background'])
            directions.add(m['direction'])
            populations.update(m['populations'])
            votes.update(m['votes'])
            ranks.update(m['ranks'])
            shorts.update(m['shorts'])
            areas.update(m['token_areas'])
            lengths.update(m['lengths'])
            frame_counts.add(m['frames'])
            family_counts.add(m['families'])
            distractor_counts.add(m['distractors'])
            changes['pairs'] += 1
        # Metamorphic probes at 42 inputs: geometry and palette are visible.
        pair = puzzle['test'][0]
        palette = np.array([7, 2, 9, 0, 4, 8, 1, 6, 3, 5], dtype=np.int8)
        transformed = palette[np.rot90(np.fliplr(pair['input']))]
        expected = palette[np.rot90(np.fliplr(pair['output']))]
        _require(np.array_equal(solve(transformed), expected), 'geometric/palette equivariance')
    _require(len(signatures) == 42, 'puzzle diversity')
    _require(len(dimensions) >= 25 and len(masks) >= 60, 'structural diversity')
    _require(backgrounds == set(range(10)), 'background roles')
    _require(directions == {'up', 'down', 'left', 'right'}, 'orientation coverage')
    _require(populations == {2, 3, 4} and ranks == {1, 2, 3, 4}, 'rank/population coverage')
    _require(votes == {0, 1, 2, 3} and frame_counts == {2, 3, 4, 5}, 'census/geometry coverage')
    _require(family_counts == {1, 2} and shorts == {5, 7}, 'family/short-side coverage')
    _require(distractor_counts == {0, 1, 2} and len(areas) >= 12 and len(lengths) == 9,
             'object/length variation')
    # Hand-built topology intervention: remove the wall between two chambers
    # and add a tail cell. Color, component count and colored area stay fixed;
    # one two-hole ballot becomes one one-hole ballot, changing its destination.
    probe = np.zeros((19, 28), dtype=np.int8)
    frame_only = probe.copy()
    for box in ((0, 0, 5, 9), (6, 0, 5, 17), (12, 0, 5, 13)):
        _draw_frame(probe, box, 4)
        _draw_frame(frame_only, box, 4)
    probe[1:4, 20:25] = 4*_ring([1, 1])
    before_expected = frame_only.copy()
    before_expected[14, 10] = 4
    _require(np.array_equal(solve(probe), before_expected), 'two-hole manual ballot')
    changed = probe.copy()
    changed[2, 22], changed[1, 25] = 0, 4
    after_expected = frame_only.copy()
    after_expected[8, 14] = 4
    _require(np.count_nonzero(changed) == np.count_nonzero(probe), 'probe area changed')
    _require(np.array_equal(solve(changed), after_expected), 'topology must change address')
    _require(not np.array_equal(before_expected, after_expected), 'topology probe is vacuous')
    # Reproducibility and count validation do not retain more than two puzzles.
    a, b = generate_puzzle(17, 10, 4), generate_puzzle(17, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(0, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(0)
    pair = bad['test'][0]
    r, c = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][r, c] = pair['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    return {'seeds': 42, 'pairs': changes['pairs'], 'distinct_puzzles': len(signatures),
            'shapes': len(dimensions), 'token_masks': len(masks), 'token_areas': sorted(areas),
            'backgrounds': sorted(backgrounds), 'directions': sorted(directions),
            'populations': sorted(populations), 'ranks': sorted(ranks), 'votes': sorted(votes),
            'frame_counts': sorted(frame_counts), 'frame_lengths': sorted(lengths),
            'topology_intervention': 'passed'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
