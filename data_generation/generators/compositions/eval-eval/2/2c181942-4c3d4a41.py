"""Assemble a color-labelled compound key; push pins against its upper envelope.

All semantic reading uses input pixels. No parent module, examples, sampled
orientation, seed, or output participates in solve(). See task_spec.md.
"""
from itertools import combinations
import numpy as np


SIDES = range(4)  # top, right, bottom, left
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int64, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    found = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, cells = [first], [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        found.append(np.asarray(sorted(cells), dtype=int))
    return found


def _attachments(mask):
    results = {}
    for turn in range(4):
        a = np.rot90(mask, turn).copy()
        h, w = a.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(a[-1], edge):
            key = _key(a)
            if key not in results:
                results[key] = (a, [])
            results[key][1].append(turn)
    return list(results.values())


def _hub_cells(r, c, labels):
    result = {}
    for side, cells in enumerate(([(r, c + 1), (r, c + 2)],
                                  [(r + 1, c + 3), (r + 2, c + 3)],
                                  [(r + 3, c + 1), (r + 3, c + 2)],
                                  [(r + 1, c), (r + 2, c)])):
        result.update((p, labels[side]) for p in cells)
    return result


def _docked(mask, side, r, c):
    marker = np.zeros_like(mask)
    marker[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    target = ((r - 1, c + 1), (r + 1, c + 4),
              (r + 4, c + 1), (r + 1, c - 1))[side]
    oriented = np.rot90(mask, -side)
    edge = np.argwhere(np.rot90(marker, -side))
    offset = np.asarray(target) - edge.min(axis=0)
    return np.argwhere(oriented) + offset


def _frame(a, bg):
    runs = {}
    for r, row in enumerate(a):
        endpoints = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), len(row)]
        for l, stop in zip(endpoints, endpoints[1:]):
            if stop - l >= 10 and row[l] != bg:
                runs.setdefault((int(row[l]), l, stop - 1), []).append(r)
    frames = []
    for (color, l, right), rows in runs.items():
        for top, bottom in combinations(rows, 2):
            if bottom - top >= 9 and np.all(a[top:bottom + 1, l] == color) and np.all(a[top:bottom + 1, right] == color):
                frames.append((top, bottom, l, right, color))
    _require(len(frames) == 1, "one closed rectangular housing is required")
    return frames[0]


def _find_hub(a, bg, frame):
    top, bottom, left, right, structure = frame
    occupied = np.zeros((4, 4), dtype=bool)
    occupied[0, 1:3] = occupied[3, 1:3] = True
    occupied[1:3, 0] = occupied[1:3, 3] = True
    hubs = []
    for r in range(top + 2, bottom - 3):
        for c in range(left + 1, right - 3):
            b = a[r:r + 4, c:c + 4]
            if np.any(b[~occupied] != bg):
                continue
            edges = (b[0, 1:3], b[1:3, 3], b[3, 1:3], b[1:3, 0])
            labels = tuple(int(v[0]) for v in edges)
            if len(set(labels)) == 4 and not set(labels) & {bg, structure} and all(v[0] == v[1] for v in edges):
                hubs.append((r, c, labels))
    _require(len(hubs) == 1, "one hollow four-label hub is required")
    return hubs[0]


def _skyline(cells):
    sky = {}
    for r, c in cells:
        sky[c] = min(sky.get(c, r), r)
    return sky


def _read(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > a.size // 2, "background must be the unique majority")
    frame = _frame(a, bg)
    top, bottom, left, right, structure = frame
    r, c, labels = _find_hub(a, bg, frame)
    permanent = np.full_like(a, bg)
    permanent[top, left:right + 1] = permanent[bottom, left:right + 1] = structure
    permanent[top:bottom + 1, left] = permanent[top:bottom + 1, right] = structure
    hub = _hub_cells(r, c, labels)
    for p, color in hub.items():
        permanent[p] = color
    reconstruction = permanent.copy()
    columns = np.flatnonzero(a[top + 1, left + 1:right] != bg) + left + 1
    _require(len(columns) >= 1 and np.all(np.diff(columns) >= 2), "pins must occupy separated lanes")
    pins = []
    for col in columns:
        col = int(col)
        end = top + 1
        while end < bottom and a[end, col] != bg:
            end += 1
        seq = tuple(map(int, a[top + 1:end, col]))
        _require(seq and not set(seq) & set(labels), "pin colors must be distinct from hub labels")
        _require(not (c <= col < c + 4 and end > r), "input pin intersects the hub bounding box")
        reconstruction[top + 1:end, col] = seq
        pins.append(dict(column=col, sequence=seq))
    outside = a != bg
    outside[top:bottom + 1, left:right + 1] = False
    pieces = []
    assembly = dict(hub)
    used = set()
    # Outside components are segmented per color, with separation checked below.
    for color in sorted(set(map(int, a[outside]))):
        for cells in _components(outside & (a == color)):
            for rr, cc in cells:
                _require(not (top - 1 <= rr <= bottom + 1 and left - 1 <= cc <= right + 1), "piece touches housing")
            lo, hi = cells.min(axis=0), cells.max(axis=0)
            mask = np.zeros(tuple(hi - lo + 1), dtype=bool)
            mask[tuple((cells - lo).T)] = True
            piece = dict(color=color, cells=cells, mask=mask, side=None)
            reconstruction[tuple(cells.T)] = color
            if color in labels:
                side = labels.index(color)
                _require(side not in used, "duplicate matching label")
                used.add(side)
                choices = _attachments(mask)
                _require(len(choices) == 1, "ambiguous or missing attachment edge")
                canonical, turns = choices[0]
                dest = _docked(canonical, side, r, c)
                for rr, cc in dest:
                    p = (int(rr), int(cc))
                    _require(top + 1 < rr < bottom and left < cc < right, "assembled piece touches a wall")
                    _require(p not in assembly, "assembled pieces overlap")
                    assembly[p] = color
                piece.update(side=side, canonical=canonical, target=dest,
                             turns={(t - side) % 4 for t in turns})
            pieces.append(piece)
    for i, p in enumerate(pieces):
        expanded = {(int(rr) + dr, int(cc) + dc) for rr, cc in p['cells'] for dr, dc in (*NEIGHBORS, (0, 0))}
        for q in pieces[i + 1:]:
            _require(not expanded & set(map(tuple, q['cells'])), "loose pieces touch each other")
    _require(np.array_equal(a, reconstruction), "extra cells or malformed pins inside housing")
    sky, hub_sky = _skyline(assembly), _skyline(hub)
    _require(all(v >= top + 2 for v in sky.values()), "key must leave a surviving pin cell below roof")
    for pin in pins:
        col, seq = pin['column'], pin['sequence']
        contact = sky.get(col)
        d = 0 if contact is None else max(0, top + len(seq) - contact + 1)
        hd = 0 if col not in hub_sky else max(0, top + len(seq) - hub_sky[col] + 1)
        _require(d < len(seq), "entire pin would be clipped")
        pin.update(shift=d, hub_only_shift=hd, skyline=contact)
    return dict(grid=a, background=bg, frame=frame, hub=(r, c, labels),
                permanent=permanent, assembly=assembly, pieces=pieces, pins=pins)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution: geometry, assembly union, contact, ordered clipping."""
    f = _read(input_grid)
    answer = f['permanent'].copy()
    for p, color in f['assembly'].items():
        answer[p] = color
    roof = f['frame'][0]
    for pin in f['pins']:
        for index, color in enumerate(pin['sequence']):
            row = roof + 1 + index - pin['shift']
            if row > roof:
                _require(answer[row, pin['column']] == f['background'], "pin/key collision after displacement")
                answer[row, pin['column']] = color
    return answer.astype(np.int8)


def _cam(rng, width, tall=False):
    for _ in range(100):
        peak = 4 if tall else int(rng.integers(2, 4))
        stem = 2 if tall else int(rng.integers(1, min(2, 5 - peak) + 1))
        mode = int(rng.integers(5))
        profile = rng.integers(1, peak + 1, width)
        profile[int(rng.integers(width))] = peak
        if mode == 0:
            profile[:] = peak
        elif mode in (1, 2):
            profile.sort()
            if mode == 2:
                profile = profile[::-1]
        elif mode == 3:
            profile[::2], profile[1::2] = peak, 1
        else:
            profile[0] = profile[-1] = peak  # separated equal maxima
        a = np.zeros((peak + stem, width), dtype=bool)
        for col, height in enumerate(profile):
            a[peak - height:peak, col] = True
        a[peak:, width // 2 - 1:width // 2 + 1] = True
        if len(_attachments(a)) == 1:
            return a
    # This flat shoulder has exactly one centered extreme attachment.
    a = np.ones((6 if tall else 4, width), dtype=bool)
    a[-2:] = False
    a[-2:, width // 2 - 1:width // 2 + 1] = True
    return a


def _draw_core(rng, labels, structure, bg, pieces, witness):
    cells = _hub_cells(0, 0, labels)
    for p in pieces:
        if p['side'] is None:
            continue
        target = _docked(p['canonical'], p['side'], 0, 0)
        if any(tuple(x) in cells for x in target):
            return None
        cells.update((tuple(x), p['color']) for x in target)
    coords = np.asarray(list(cells))
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    roof_gap = 3  # at least two cells of a roof-clipped suffix can survive
    frame_lo = lo - (roof_gap, 2)
    frame_hi = hi + (2, 2)
    shape = tuple(frame_hi - frame_lo + 1)
    if max(shape) > 20:
        return None
    hr, hc = map(int, -frame_lo)
    h, w = map(int, shape)
    assembly = {(int(rr - frame_lo[0]), int(cc - frame_lo[1])): color for (rr, cc), color in cells.items()}
    sky = _skyline(assembly)
    core = np.full((h, w), bg, dtype=np.int8)
    core[0] = core[-1] = structure
    core[:, 0] = core[:, -1] = structure
    for point, color in _hub_cells(hr, hc, labels).items():
        core[point] = color
    max_length = {col: (hr - 1 if hc <= col < hc + 4 else h - 2) for col in range(1, w - 1)}
    forced = {}
    if witness:
        # {top,left}: shared col hc-1 is shielded by the top part. Outside
        # lane hc-3 meets the left part at a different height. Equal original
        # lengths isolate the union skyline's effect, including roof clipping.
        x, y, gap = hc - 3, hc - 1, hc + 1
        if x not in sky or y not in sky or sky[x] == sky[y]:
            return None
        length = max(sky[x], sky[y]) + 1
        if length > min(max_length[x], max_length[y]):
            return None
        forced[x] = forced[y] = length
        forced[gap] = max(1, sky[gap] - 2)
    else:
        eligible = [col for col in sky if max_length[col] >= sky[col]]
        if eligible:
            col = int(rng.choice(eligible))
            forced[col] = int(rng.integers(sky[col], max_length[col] + 1))
    cols = sorted(forced)
    desired = int(rng.integers(max(3, len(cols)), min(7, (w - 1) // 2) + 1))
    options = list(map(int, rng.permutation(np.arange(1, w - 1))))
    for col in options:
        if len(cols) >= desired:
            break
        if all(abs(col - other) >= 2 for other in cols):
            cols.append(col)
    cols.sort()
    allowed = [v for v in range(10) if v not in (bg, *labels)]
    payload = [v for v in allowed if v != structure]
    for col in cols:
        limit = max_length[col]
        s = sky.get(col)
        if col in forced:
            length = forced[col]
        elif s is None:
            length = int(rng.integers(1, limit + 1))
        else:
            contact = s - 1
            kind = int(rng.integers(3))
            if kind == 0 and contact <= limit:
                length = contact
            elif kind == 1 and s <= limit:
                length = int(rng.integers(s, limit + 1))
            else:
                length = int(rng.integers(1, min(limit, max(1, contact - 1)) + 1))
        if witness and col in (hc - 3, hc - 1):
            # Ordered suffix after clipping has at least two distinct colors.
            seq = [payload[i % len(payload)] for i in range(length)]
        elif rng.random() < .45:
            color = int(rng.choice(payload))
            prefix = int(rng.integers(0, length))
            seq = [structure] * prefix + [color] * (length - prefix)
        elif rng.random() < .5:
            seq = [int(rng.choice(payload))] * length
        else:
            seq = list(map(int, rng.choice(allowed, length)))
        core[1:1 + length, col] = seq
    return core


def _pack(rng, core, bg, pieces):
    masks = [np.rot90(p['canonical'], p['input_turns']) for p in pieces]
    fh, fw = core.shape
    for trial in range(12):
        h = 30 if trial >= 6 else int(rng.integers(max(24, fh + 2), 31))
        w = 30 if trial >= 6 else int(rng.integers(max(24, fw + 2), 31))
        top = 0 if trial >= 6 else int(rng.integers(0, min(4, h - fh)))
        left = 0 if trial >= 6 else int(rng.integers(0, min(4, w - fw)))
        a = np.full((h, w), bg, dtype=np.int8)
        a[top:top + fh, left:left + fw] = core
        blocked = np.zeros_like(a, dtype=bool)
        blocked[max(0, top - 1):top + fh + 1, max(0, left - 1):left + fw + 1] = True
        order = list(map(int, rng.permutation(len(pieces))))
        order.sort(key=lambda i: -masks[i].size)
        ok = True
        for i in order:
            mask = masks[i]
            mh, mw = mask.shape
            local = np.argwhere(mask)
            tr, tc = np.indices((h - mh + 1, w - mw + 1))
            rr = tr.ravel()[:, None] + local[:, 0]
            cc = tc.ravel()[:, None] + local[:, 1]
            good = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
            if not len(good):
                ok = False
                break
            index = int(rng.choice(good))
            a[rr[index], cc[index]] = pieces[i]['color']
            for r, c in zip(rr[index], cc[index]):
                blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        if ok and np.count_nonzero(a == bg) > a.size // 2:
            return a
    return _pack_strips(core, bg, pieces, masks)


def _pack_strips(core, bg, pieces, masks):
    fh, fw = core.shape
    # Enumerate a finite partition between right and bottom strips. Every
    # generated mask fits 8x8 and has at least one dimension <=6. Keep the
    # bottom strip's objects left of col 21, avoiding the strips' shared corner.
    for bits in range(1 << len(pieces)):
        bottom = [i for i in range(len(pieces)) if bits & (1 << i)]
        right = [i for i in range(len(pieces)) if i not in bottom]
        if sum(masks[i].shape[1] + 1 for i in bottom) > 21:
            continue
        if sum(masks[i].shape[0] + 1 for i in right) > 31:
            continue
        a = np.full((30, 30), bg, dtype=np.int8)
        a[:fh, :fw] = core
        row = 0
        for i in right:
            rr, cc = np.nonzero(masks[i])
            a[rr + row, cc + 21] = pieces[i]['color']
            row += masks[i].shape[0] + 1
        col = 0
        for i in bottom:
            rr, cc = np.nonzero(masks[i])
            a[rr + 21, cc + col] = pieces[i]['color']
            col += masks[i].shape[1] + 1
        return a
    raise RuntimeError("could not pack six bounded pieces")


def _make_pair(rng, sides, rotations, *, witness=False, special=False, test=False):
    for attempt in range(60):
        palette = list(map(int, rng.permutation(10)))
        bg, structure, labels = palette[0], palette[1], tuple(palette[2:6])
        tall_side = int(rng.choice(sorted(sides))) if test else None
        pieces = []
        for side in sorted(sides):
            width = 8 if side == tall_side else (6 if special else int(rng.choice((4, 6))))
            mask = _cam(rng, width, side == tall_side)
            delta = rotations.get(side, int(rng.integers(4)))
            pieces.append(dict(color=labels[side], side=side, canonical=mask,
                               input_turns=(-side - delta) % 4))
        nd = 1 if special else int(rng.integers(3))
        for j in range(nd):
            if j == 0 or rng.random() < .5:
                mask = _cam(rng, int(rng.choice((4, 6))))
            else:
                mask = np.asarray([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
            pieces.append(dict(color=int(rng.choice(palette[6:])), side=None,
                               canonical=mask, input_turns=int(rng.integers(4))))
        core = _draw_core(rng, labels, structure, bg, pieces, witness)
        if core is None:
            continue
        a = _pack(rng, core, bg, pieces)
        # Generation acceptance reconstructs completed pixels, never sampled
        # orientation. The parser is not restricted to the cam shape catalog.
        try:
            f = _read(a)
            if test and not any(p['shift'] for p in f['pins']):
                continue
            return dict(input=a, output=solve(a))
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted")


def _counts(nt, nq):
    for value, low, high, name in ((nt, 2, 10, 'num_train'), (nq, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    turns = list(map(int, rng.permutation(4)))
    configs = [({0, 3}, {0: turns[0], 3: turns[3]}, True, True),
               ({1, 2}, {1: turns[1], 2: turns[2]}, False, True)]
    for _ in range(num_train - 2):
        n = int(rng.integers(1, 5))
        configs.append((set(map(int, rng.choice(4, n, replace=False))), {}, False, False))
    result, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                sides, rotations, witness, special = configs[i]
            else:
                n = int(rng.integers(2, 5))
                sides = set(map(int, rng.choice(4, n, replace=False)))
                rotations, witness, special = {}, False, False
            for _ in range(10):
                pair = _make_pair(rng, sides, rotations, witness=witness,
                                  special=special, test=split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct input")
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def _features(f):
    matched = [p for p in f['pieces'] if p['side'] is not None]
    pins = f['pins']
    projections = {}
    for p in matched:
        for col in set(map(int, p['target'][:, 1])):
            projections[col] = projections.get(col, 0) + 1
    return dict(sides={p['side'] for p in matched},
                turns=set().union(*(p['turns'] for p in matched)) if matched else set(),
                distractor=any(p['side'] is None and len(_attachments(p['mask'])) == 1 for p in f['pieces']),
                competition=any(projections.get(p['column'], 0) >= 2 and p['shift'] > 0 for p in pins),
                a_matters=any(p['shift'] != p['hub_only_shift'] for p in pins),
                b_matters=any(p['shift'] > 0 for p in pins),
                gap=any(p['skyline'] is not None and f['frame'][0] + len(p['sequence']) < p['skyline'] - 1 for p in pins),
                ordered=any(p['shift'] > 0 and len(set(p['sequence'][p['shift']:])) >= 2 for p in pins),
                equal_length_unequal_motion=any(len(p['sequence']) == len(q['sequence']) and p['shift'] != q['shift']
                                              for p, q in combinations(pins, 2)),
                matched=matched)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "shape changed")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _read(a)
            _require(np.array_equal(solve(a), b), "incorrect output")
            top, bottom, left, right, _ = f['frame']
            _require(np.all(b[top, left:right + 1] == a[top, left:right + 1]) and
                     np.all(b[bottom, left:right + 1] == a[bottom, left:right + 1]) and
                     np.all(b[top:bottom + 1, (left, right)] == a[top:bottom + 1, (left, right)]), "frame changed")
            for point, color in f['assembly'].items():
                _require(b[point] == color, "shape lost or recolored")
            for p in f['pieces']:
                _require(np.all(b[tuple(p['cells'].T)] == f['background']), "loose original not erased")
            for pin in f['pins']:
                seq = pin['sequence'][pin['shift']:]
                _require(tuple(map(int, b[top + 1:top + 1 + len(seq), pin['column']])) == seq, "sequence suffix changed")
            features = _features(f)
            if split == 'train':
                train.append(features)
    _require(set().union(*(d['sides'] for d in train)) == set(SIDES), "training misses a docking side")
    _require(set().union(*(d['turns'] for d in train)) == set(SIDES), "training misses a required rotation")
    for feature in ('distractor', 'competition', 'a_matters', 'b_matters', 'gap', 'ordered', 'equal_length_unequal_motion'):
        _require(any(d[feature] for d in train), f"training misses {feature} witness")


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _literal_regression():
    """A manually specified scene and answer, not made by the sampler."""
    a = np.zeros((26, 26), dtype=np.int8)
    a[0, :18] = a[17, :18] = 9
    a[:18, 0] = a[:18, 17] = 9
    for point, color in _hub_cells(9, 8, (1, 2, 3, 4)).items():
        a[point] = color
    sequences = {5: [9, 5, 6, 5, 6, 5, 6, 5, 6, 5],
                 7: [9, 9, 5, 6, 5, 6, 5, 6, 5, 6],
                 9: [5] * 5, 12: [6] * 6, 15: [7, 8, 7, 8]}
    for col, seq in sequences.items():
        a[1:1 + len(seq), col] = seq
    top = np.asarray([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0]], dtype=bool)
    left = np.asarray([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0, 0]], dtype=bool)
    for mask, turn, row, col, color in ((top, 1, 19, 1, 1),
                                        (left, 2, 19, 6, 4), (top, 0, 19, 14, 8)):
        rr, cc = np.nonzero(np.rot90(mask, turn))
        a[rr + row, cc + col] = color
    expected = np.zeros_like(a)
    expected[0, :18] = expected[17, :18] = 9
    expected[:18, 0] = expected[:18, 17] = 9
    for point, color in _hub_cells(9, 8, (1, 2, 3, 4)).items():
        expected[point] = color
    # Literal placed key pixels: top flange shields the left stem in col 7.
    expected[7, 7:13] = 1
    expected[8, 9:11] = 1
    expected[8:14, 5] = 4
    expected[10:12, 6:8] = 4
    # Equal length ten: shifts three and four. A gap, exact contact, and a
    # missed projection stay unchanged. Leading structural cells are lost.
    expected[1:8, 5] = [5, 6, 5, 6, 5, 6, 5]
    expected[1:7, 7] = [5, 6, 5, 6, 5, 6]
    expected[1:6, 9] = 5
    expected[1:7, 12] = 6
    expected[1:5, 15] = [7, 8, 7, 8]
    assert np.array_equal(solve(a), expected)
    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(expected))
    f = _read(a)
    assert [p['shift'] for p in f['pins']] == [3, 4, 0, 0, 0]
    assert _features(f)['competition'] and _features(f)['a_matters']
    # A different two-cell stem depth is supported without knowing a seed.
    assert len(_attachments(left)) == 1
    # An eight-connected diagonal silhouette also has a valid attachment.
    # Narrowing the upper piece releases the shared pin: it now hits the
    # lower stem, loses only one cell, and retains a structural prefix.
    diagonal = np.asarray([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)
    b = a.copy()
    b[19:25, 1:3] = 0
    rr, cc = np.nonzero(np.rot90(diagonal))
    b[rr + 19, cc + 1] = 1
    diagonal_expected = expected.copy()
    diagonal_expected[7, 7:13] = 0
    diagonal_expected[7, (8, 11)] = 1
    diagonal_expected[1:10, 7] = [9, 5, 6, 5, 6, 5, 6, 5, 6]
    assert np.array_equal(solve(b), diagonal_expected)
    return a, expected


def self_test():
    """Bounded semantic regressions, 42-seed coverage and explicit count checks."""
    _literal_regression()
    # Exhaust the seven worst-case six-piece orientation mixtures in the
    # finite packing fallback, checking conservation and separation. Smaller
    # generated bounding boxes are dominated by these 8x6 / 6x8 rectangles.
    core = np.zeros((20, 20), dtype=np.int8)
    core[0] = core[-1] = 9
    core[:, 0] = core[:, -1] = 9
    pieces = [dict(color=i + 1) for i in range(6)]
    for portraits in range(7):
        masks = [np.ones((8, 6) if i < portraits else (6, 8), dtype=bool) for i in range(6)]
        packed = _pack_strips(core, 0, pieces, masks)
        assert np.array_equal(packed[:20, :20], core)
        assert np.count_nonzero(packed == 0) > packed.size // 2
        occupied = []
        for p in pieces:
            components = _components(packed == p['color'])
            assert len(components) == 1 and len(components[0]) == 48
            occupied.append(set(map(tuple, components[0])))
        for i, points in enumerate(occupied):
            expanded = {(r + dr, c + dc) for r, c in points for dr, dc in NEIGHBORS}
            assert all(not expanded & q for q in occupied[i + 1:])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, palettes, silhouettes, subsets, counts, profiles, depths, widths = (set() for _ in range(8))
    contact = gap = misses = multicolor = 0
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _read(pair['input'])
                d = _features(f)
                shapes.add(pair['input'].shape)
                palettes.add((f['background'], f['frame'][-1], f['hub'][2]))
                subsets.add(tuple(sorted(d['sides'])))
                counts.add(len(f['pins']))
                for p in d['matched']:
                    m = p['canonical']
                    silhouettes.add(_key(m))
                    depths.add(m.shape[0])
                    widths.add(m.shape[1])
                    profiles.add(tuple(np.argmax(m, axis=0)))
                for p in f['pins']:
                    top = f['frame'][0]
                    contact += p['skyline'] is not None and top + len(p['sequence']) == p['skyline'] - 1
                    gap += p['skyline'] is not None and top + len(p['sequence']) < p['skyline'] - 1
                    misses += p['skyline'] is None
                    multicolor += len(set(p['sequence'][p['shift']:])) >= 2
    assert len(shapes) >= 20 and len(palettes) >= 100
    assert len(silhouettes) >= 60 and len(profiles) >= 40
    assert widths == {4, 6, 8} and {3, 4, 5, 6} <= depths
    assert len(subsets) >= 12 and len(counts) >= 4
    assert min(contact, gap, misses, multicolor) > 10
    assert _puzzle_key(generate_puzzle(991)) == _puzzle_key(generate_puzzle(991))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(2000 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Input-only solver covariance under arbitrary palette permutations.
    p = puzzles[0]['train'][0]
    mapping = np.asarray([3, 7, 1, 9, 0, 6, 2, 8, 5, 4])
    assert np.array_equal(solve(mapping[p['input']]), mapping[p['output']])
    # Assembly envelope is independent of the order in which parts are read.
    f = _read(puzzles[0]['train'][0]['input'])
    reverse_union = _hub_cells(*f['hub'])
    for part in reversed(f['pieces']):
        if part['side'] is not None:
            reverse_union.update((tuple(x), part['color']) for x in part['target'])
    assert reverse_union == f['assembly']
    assert len(_attachments(np.ones((2, 2), dtype=bool))) == 1
    malformed = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    bad = malformed['test'][0]['output']
    bad[0, 0] = (int(bad[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return dict(seeds=42, distinct_shapes=len(shapes), silhouettes=len(silhouettes),
                profiles=len(profiles), side_subsets=len(subsets), pin_counts=sorted(counts),
                contact=contact, gap=gap, misses=misses, surviving_multicolor=multicolor)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
