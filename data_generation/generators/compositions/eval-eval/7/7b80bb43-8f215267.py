"""Circuit census: repair wire topology, then compress its matching cargo.

Standard library + NumPy only. No files, parent imports, random globals, or
generation state are consulted by solve(). Diagnostic sweeps stream in memory.
"""

import hashlib
import numpy as np


STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))
CHECK_SEEDS = (0, 47116864, 389252222, 347964490, 468413414, 556408082,
               509752628, 944071616, 548067430, 791682901, 68897479, 972244275,
               835087939, 538743204, 982310162, 461911969, 825375545, 143570944,
               315735918, 101313548, 638558872, 752686058, 440640303, 767374364,
               184596108, 535918210, 574924813, 529276759, 252581220, 32211034,
               577210934, 778504277, 927256869, 917555169, 726237356, 341419540,
               67034510, 575171493, 455898706, 599434947, 522908095, 647389868)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _counts(seed, num_train, num_test):
    # Explicit None also has a deterministic useful interpretation. Omitted
    # arguments keep the requested public defaults of four and one.
    n = None if num_train is None else _integer(num_train, 'num_train')
    t = None if num_test is None else _integer(num_test, 'num_test')
    if n is not None and not 2 <= n <= 10:
        raise ValueError('num_train must be in 2..10')
    if t is not None and not 1 <= t <= 4:
        raise ValueError('num_test must be in 1..4')
    if n is None and t is None:
        t = 1 + seed % 4
        n = t + 1 + (seed // 4) % (10 - t)
    elif n is None:
        n = t + 1 + seed % (10 - t)
    elif t is None:
        t = 1 + seed % min(4, n - 1)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    return n, t


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), 'grid dimensions')
    _check(np.issubdtype(a.dtype, np.integer), 'grid must be integral')
    _check(bool(((0 <= a) & (a <= 9)).all()), 'grid colors')
    return a.astype(np.int8, copy=True)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _halo(mask):
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    parts = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if labels[start] >= 0:
            continue
        labels[start] = len(parts)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if _at(mask, nr, nc) and labels[nr, nc] < 0:
                    labels[nr, nc] = len(parts)
                    stack.append((nr, nc))
        parts.append(tuple(sorted(cells)))
    return labels, parts


def _strong(mask):
    result = np.zeros_like(mask)
    for m, out in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for i in range(3):
            out[:, i:i+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        candidates = m[:, :-4] & m[:, 4:]
        candidates &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(candidates):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repair(mask):
    """Simultaneous input-evidenced repair; never closes an unmarked gap."""
    strong = _strong(mask)
    additions = np.zeros_like(mask)
    anchors = np.zeros_like(mask)
    records = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        cues = []
        for end in (0, 4):
            ar, ac = r+end*dr, c+end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in spur):
                    if not _at(mask, ar+3*sr, ac+3*sc):
                        cues.append(spur)
        if cues:
            _check(len(cues) == 1, 'ambiguous repair cue')
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            for p in gap:
                additions[p] = True
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
            records.append((gap, cues[0]))
    weak = mask & ~_strong(mask | additions) & ~anchors
    debris = np.zeros_like(mask)
    for dc in (-1, 1):
        if dc == 1:
            linked = weak[:-1, :-1] & weak[1:, 1:]
            debris[:-1, :-1] |= linked
            debris[1:, 1:] |= linked
        else:
            linked = weak[:-1, 1:] & weak[1:, :-1]
            debris[:-1, 1:] |= linked
            debris[1:, :-1] |= linked
    return (mask & ~debris) | additions, records


def _colored_parts(grid, bg, omit=None):
    parts = []
    omit = np.zeros(grid.shape, bool) if omit is None else omit
    for color in sorted(int(c) for c in np.unique(grid) if c != bg):
        _, groups = _components((grid == color) & ~omit)
        for cells in groups:
            rs, cs = zip(*cells)
            parts.append(dict(color=color, cells=cells,
                              box=(min(rs), max(rs), min(cs), max(cs))))
    return parts


def _is_frame(part):
    r0, r1, c0, c1 = part['box']
    h, w = r1-r0+1, c1-c0+1
    if min(h, w) not in (5, 7) or max(h, w) not in (7, 9, 11, 13) or h == w:
        return False
    perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                 if r in (r0, r1) or c in (c0, c1)}
    return set(part['cells']) == perimeter


def _contacts(cells, labels):
    ids = set()
    edges = 0
    for r, c in cells:
        for dr, dc in STEPS:
            rr, cc = r+dr, c+dc
            if 0 <= rr < labels.shape[0] and 0 <= cc < labels.shape[1] and labels[rr, cc] >= 0:
                ids.add(int(labels[rr, cc]))
                edges += 1
    return ids, edges


def _census(frames, objects, wire):
    labels, groups = _components(wire)
    counts, areas = [], []
    touched = [_contacts(o['cells'], labels)[0] for o in objects]
    for f in frames:
        identity = int(labels[f['port']])
        _check(identity >= 0, 'port lost its wire')
        selected = [o for o, ids in zip(objects, touched)
                    if o['color'] == f['color'] and identity in ids]
        counts.append(len(selected))
        areas.append(sum(len(o['cells']) for o in selected))
    return counts, areas, labels, groups, touched


def _inspect(grid):
    colors, freqs = np.unique(grid, return_counts=True)
    _check(np.count_nonzero(freqs == freqs.max()) == 1, 'background is not unique')
    bg = int(colors[np.argmax(freqs)])
    frames = [p for p in _colored_parts(grid, bg) if _is_frame(p)]
    _check(1 <= len(frames) <= 3, 'expected one through three terminals')
    frame_mask = np.zeros(grid.shape, bool)
    wire_colors = set()
    for f in frames:
        r0, r1, c0, c1 = f['box']
        _check(bool((grid[r0+1:r1, c0+1:c1] == bg).all()), 'input terminal not hollow')
        if c1-c0 > r1-r0:
            options = [((r0+r1)//2, c0-1, (0, -1)),
                       ((r0+r1)//2, c1+1, (0, 1))]
        else:
            options = [(r0-1, (c0+c1)//2, (-1, 0)),
                       (r1+1, (c0+c1)//2, (1, 0))]
        ports = [(r, c, direction) for r, c, direction in options
                 if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != bg]
        _check(len(ports) == 1, 'terminal needs one midpoint port')
        r, c, direction = ports[0]
        f['port'], f['direction'] = (r, c), direction
        wire_colors.add(int(grid[r, c]))
        for p in f['cells']:
            frame_mask[p] = True
        _check(int(grid[r, c]) != f['color'], 'wire and category coincide')
    _check(len(wire_colors) == 1, 'wire color inconsistent')
    line = wire_colors.pop()
    wire_input = grid == line
    clean, repairs = _repair(wire_input)
    objects = _colored_parts(grid, bg, frame_mask | wire_input)
    counts, areas, labels, groups, touched = _census(frames, objects, clean)
    _check(len({int(labels[f['port']]) for f in frames}) == len(frames),
           'multiple terminals on one network')
    for o, ids in zip(objects, touched):
        _check(1 <= len(o['cells']) <= 7 and len(ids) <= 1, 'ambiguous/oversized cargo')
    capacities = [(max(f['box'][1]-f['box'][0], f['box'][3]-f['box'][2])-1)//2 for f in frames]
    _check(all(n <= cap for n, cap in zip(counts, capacities)), 'tally overflow')
    raw_counts = _census(frames, objects, clean & wire_input)[0]
    global_counts = [sum(o['color'] == f['color'] for o in objects) for f in frames]
    all_filled = clean.copy()
    for r, c, dr, dc in _gaps(clean):
        for k in (1, 2, 3):
            all_filled[r+k*dr, c+k*dc] = True
    filled_counts = _census(frames, objects, all_filled)[0]
    return dict(bg=bg, line=line, frames=frames, frame_mask=frame_mask,
                wire_input=wire_input, clean=clean, repairs=repairs, objects=objects,
                counts=counts, areas=areas, labels=labels, groups=groups, touched=touched,
                raw_counts=raw_counts, global_counts=global_counts, filled_counts=filled_counts)


def _render(info, counts=None):
    result = np.full(info['clean'].shape, info['bg'], np.int8)
    result[info['clean']] = info['line']
    counts = info['counts'] if counts is None else counts
    for f, count in zip(info['frames'], counts):
        for p in f['cells']:
            result[p] = f['color']
        dr, dc = f['direction']
        pr, pc = f['port']
        # Port is one cell outside the outline, so subtract one then pitch two.
        for k in range(1, count+1):
            result[pr-(2*k+1)*dr, pc-(2*k+1)*dc] = f['color']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, repairs, owners and tallies from this input alone."""
    return _render(_inspect(_grid(input_grid)))


def _shape(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS} - cells)
        # Keep the independent cargo sampler compact enough for narrow regions.
        frontier = [p for p in frontier
                    if max(r for r, c in cells | {p}) - min(r for r, c in cells | {p}) < 3
                    and max(c for r, c in cells | {p}) - min(c for r, c in cells | {p}) < 4]
        cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _draw_frame(grid, box, color):
    r0, r1, c0, c1 = box
    grid[r0, c0:c1+1] = grid[r1, c0:c1+1] = color
    grid[r0:r1+1, c0] = grid[r0:r1+1, c1] = color


def _damage_choices(clean):
    """Corrupt only a visible straight span; accept exact geometric recovery."""
    choices = []
    for r, c in np.argwhere(clean):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            if not all(_at(clean, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(clean, r-dr, c-dc) or _at(clean, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(_at(clean, rr+dc, cc+dr) or _at(clean, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for end in (0, 4):
                ar, ac = r+end*dr, c+end*dc
                toward = 1 if end == 0 else -1
                for side in (-1, 1):
                    spur = tuple((ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2))
                    if not all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                               and not clean[rr, cc] for rr, cc in spur):
                        continue
                    trial = clean.copy()
                    for p in gap:
                        trial[p] = False
                    for p in spur:
                        trial[p] = True
                    try:
                        restored, records = _repair(trial)
                        if len(records) == 1 and np.array_equal(restored, clean):
                            choices.append((gap, spur))
                    except AssertionError:
                        pass
    return choices


def _place_object(rng, grid, shape, color, forbidden, clean, allowed_ids, labels,
                  require_contact=True, must_touch=None, min_edges=1):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    choices = []
    for r0 in range(grid.shape[0]-sh+1):
        for c0 in range(grid.shape[1]-sw+1):
            cells = tuple((r0+r, c0+c) for r, c in shape)
            if any(forbidden[p] or clean[p] for p in cells):
                continue
            ids, edges = _contacts(cells, labels)
            if require_contact and ids != allowed_ids:
                continue
            if require_contact and edges < min_edges:
                continue
            if not require_contact and ids:
                continue
            if must_touch is not None and not any(_at(must_touch, r+dr, c+dc)
                                                  for r, c in cells for dr, dc in STEPS):
                continue
            choices.append(cells)
    if not choices:
        return False
    cells = choices[int(rng.integers(len(choices)))]
    mask = np.zeros(grid.shape, bool)
    for p in cells:
        grid[p] = color
        mask[p] = True
    forbidden |= _halo(mask)
    return True


def _local_network(rng, h, w, bg, line, category, unmatched, areas,
                   bare=False, loop=False, rich=False):
    """Build from intended topology, with no use of solve for label construction."""
    fh = 7 if h >= 12 and rng.random() < .25 else 5
    lengths = [x for x in (7, 9, 11, 13) if x > fh and x >= 2*len(areas)+3]
    fw = int(rng.choice(lengths[:2]))
    r0 = int(rng.integers(h-fh+1))
    c0 = int(rng.integers(2))
    box = r0, r0+fh-1, c0, c0+fw-1
    row, start = r0+fh//2, c0+fw
    end = w-1-int(rng.integers(2))
    if end-start < (13 if bare else 7):
        return None
    clean = np.zeros((h, w), bool)
    clean[row, start:end+1] = True
    branch_end = end-7 if bare else end
    possible = list(range(start+5, branch_end+1))
    columns = []
    wanted = int(rng.integers(1, 4 if rich else 3))
    while possible and len(columns) < wanted:
        c = int(rng.choice(possible))
        columns.append(c)
        possible = [x for x in possible if abs(x-c) >= 3]
    if not columns:
        return None
    for c in columns:
        lo = int(rng.integers(max(1, row-1))) if row >= 2 else row
        hi = int(rng.integers(min(h-1, row+2), h)) if row+2 < h else row
        if rng.random() < .4:
            if row >= h//2:
                hi = row
            else:
                lo = row
        clean[lo:hi+1, c] = True
    if loop:
        # Joining two branches adds a cycle; the port still has a bridge neck.
        lo, hi = start+5, end
        level = row-3 if row >= 3 else row+3
        if hi-lo < 3 or not 0 <= level < h:
            return None
        clean[min(row, level):max(row, level)+1, lo] = True
        clean[min(row, level):max(row, level)+1, hi] = True
        clean[level, lo:hi+1] = True
    if bare:
        clean[row, end-5:end-2] = False
    if (clean[:-1, :-1] & clean[1:, :-1] & clean[:-1, 1:] & clean[1:, 1:]).any():
        return None
    try:
        if not np.array_equal(_repair(clean)[0], clean):
            return None
    except AssertionError:
        return None
    labels, groups = _components(clean)
    owner = int(labels[row, start])
    grid = np.full((h, w), bg, np.int8)
    _draw_frame(grid, box, category)
    frame_mask = grid != bg
    forbidden = _halo(frame_mask)
    grid[clean] = line
    candidates = _damage_choices(clean)
    rng.shuffle(candidates)
    chosen = None
    for gap, spur in candidates:
        if any(forbidden[p] for p in gap+spur):
            continue
        broken = clean.copy()
        for p in gap:
            broken[p] = False
        raw_labels, _ = _components(broken)
        home = int(raw_labels[row, start])
        distal = (labels == owner) & broken & (raw_labels != home)
        if distal.sum() >= 3:
            chosen = gap, spur, distal
            break
    if chosen is None:
        return None
    gap, spur, distal = chosen
    footprint = np.zeros_like(clean)
    for p in gap+spur:
        footprint[p] = True
    forbidden |= _halo(footprint)
    # Guarantee a same-colored island beyond an unmarked break in the first
    # teaching scene. It is a deliberate exclusion, not a hidden distractor tag.
    if bare:
        island = int(labels[row, end])
        if not _place_object(rng, grid, _shape(rng, int(rng.integers(1, 4))), category,
                             forbidden, clean, {island}, labels):
            return None
    # Place the teaching/count cargo. Its first object must be cut off by the
    # selected bridge when spurs are deleted without restoring the gap.
    for j, area in enumerate(areas):
        local_forbid = forbidden.copy()
        if not _place_object(rng, grid, _shape(rng, area), category, local_forbid,
                             clean, {owner}, labels, must_touch=distal if j == 0 else None,
                             min_edges=2 if bare and j == 0 else 1):
            return None
        cargo_mask = (grid != bg) & (grid != line) & ~frame_mask
        forbidden |= _halo(cargo_mask)
        if j == 0:
            _, comps = _components(cargo_mask & (grid == category))
            # Test actual contact with the distal wire, rather than bounding box.
            if not any(any(_at(distal, r+dr, c+dc) for r, c in cells for dr, dc in STEPS)
                       for cells in comps if len(cells) == area):
                return None
    # Color filtering is shown by real unmatched cargo touching the same wire.
    if not _place_object(rng, grid, _shape(rng, int(rng.integers(1, 5))), unmatched,
                         forbidden, clean, {owner}, labels):
        return None
    # Optional floating matching cargo also refutes global color counting.
    if rng.random() < .5:
        _place_object(rng, grid, _shape(rng, int(rng.integers(1, 4))), category,
                      forbidden, clean, set(), labels, require_contact=False)
    output = np.full_like(grid, bg)
    _draw_frame(output, box, category)
    output[clean] = line
    for i in range(1, len(areas)+1):
        output[row, c0+fw-1-2*i] = category
    for p in gap:
        grid[p] = bg
    for p in spur:
        grid[p] = line
    # Additional damage can lie in a branch or in a cycle, and need not alter
    # the tally. The mandatory bridge above always makes A causally relevant.
    if rich and rng.random() < .65:
        more = _damage_choices(clean)
        rng.shuffle(more)
        for gap2, spur2 in more:
            cells = gap2+spur2
            if any(forbidden[p] or _halo(footprint)[p] for p in cells):
                continue
            trial = grid.copy()
            for p in gap2:
                trial[p] = bg
            for p in spur2:
                trial[p] = line
            try:
                if np.array_equal(_repair(trial == line)[0], clean):
                    grid = trial
                    break
            except AssertionError:
                pass
    # A small bounded search adds harmless detached oblique debris.
    for _ in range(18):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dc = int(rng.choice((-1, 1)))
        length = int(rng.choice((2, 3)))
        cells = [(r+k, c+k*dc) for k in range(length)]
        occupied = _halo(grid != bg)
        if not all(0 <= rr < h and 0 <= cc < w and not occupied[rr, cc] for rr, cc in cells):
            continue
        trial = grid.copy()
        for p in cells:
            trial[p] = line
        try:
            if np.array_equal(_repair(trial == line)[0], clean):
                grid = trial
                break
        except AssertionError:
            pass
    if rng.integers(2):
        grid, output = grid[:, ::-1], output[:, ::-1]
    if rng.integers(2):
        grid, output = grid[::-1], output[::-1]
    return grid.copy(), output.copy()


def _scene(rng, categories, area_rows, bg, line, unmatched, turn,
           teaching=False, loop=False, rich=False):
    n = len(categories)
    for _ in range(48):
        if n == 3:
            heights = [9, 9, 10]
            rng.shuffle(heights)
            w, gaps = int(rng.integers(28, 31)), [1, 1]
        else:
            heights = [int(rng.integers(11, 15)) for _ in categories]
            gaps = [int(rng.integers(1, 3)) for _ in range(n-1)]
            w = int(rng.integers(27 if teaching else 24, 31))
        h = sum(heights)+sum(gaps)
        if h > 30:
            continue
        inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
        row = 0
        failed = False
        for i, (height, color, areas) in enumerate(zip(heights, categories, area_rows)):
            local = None
            for _ in range(8):
                local = _local_network(rng, height, w, bg, line, color, unmatched,
                                       areas, bare=teaching and i == 0,
                                       loop=loop, rich=rich)
                if local is not None:
                    break
            if local is None:
                failed = True
                break
            inp[row:row+height], out[row:row+height] = local
            row += height + (gaps[i] if i < n-1 else 0)
        if failed:
            continue
        inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
        try:
            info = _inspect(inp)
            if not np.array_equal(_render(info), out):
                continue
            if info['counts'] == info['raw_counts']:
                continue
            if teaching and info['counts'] == info['filled_counts']:
                continue
            return dict(input=inp, output=out)
        except AssertionError:
            continue
    raise RuntimeError('bounded circuit construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    nt, nq = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(10))
    shift, base_turn = int(rng.integers(3)), int(rng.integers(4))
    rich = bool(rng.integers(2))
    puzzle, seen = dict(train=[], test=[]), set()
    for index in range(nt+nq):
        train = index < nt
        split = 'train' if train else 'test'
        local_index = index if train else index-nt
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c != anchor])]
        bg, line, unmatched = palette[:3]
        if index == 0:
            categories, area_rows = [anchor, anchor], [[6], [1, 3]]
        elif index == 1:
            categories, area_rows = [anchor, palette[3]], [[3, 3], []]
        else:
            n = 3 if not train and shift == 0 else int(rng.integers(1, 3))
            pool = [c for c in range(10) if c not in (bg, line, unmatched)]
            categories = [int(rng.choice(pool)) for _ in range(n)]
            maximum = 2 if n == 3 else 3
            counts = [int(rng.integers(maximum+1)) for _ in range(n)]
            if not train and shift == 1:
                counts[0] = 4
            if not any(counts):
                counts[0] = 1
            area_rows = [[int(rng.integers(1, 8)) for _ in range(count)] for count in counts]
        turn = base_turn if index == 0 else (base_turn+1)%4 if index == 1 else int(rng.integers(4))
        for _ in range(8):
            pair = _scene(rng, categories, area_rows, bg, line, unmatched, turn,
                          teaching=index == 0, loop=(not train and shift == 2), rich=rich)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('could not make distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _check(inp.shape == out.shape, 'output dimensions changed')
    d = _inspect(inp)
    _check(np.array_equal(_render(d), out), 'wrong repair, census, tally, or erasure')
    clean = d['clean']
    _check(not (clean[:-1, :-1] & clean[1:, :-1] & clean[:-1, 1:] & clean[1:, 1:]).any(), 'thick wire')
    _check(bool(d['repairs']), 'no repair')
    _check(d['raw_counts'] != d['counts'], 'repair must affect a census')
    # Frames must have a moat except for their visibly encoded port.
    for f in d['frames']:
        fm = np.zeros(inp.shape, bool)
        for p in f['cells']:
            fm[p] = True
        external = _halo(fm) & ~fm
        external[f['port']] = False
        _check(bool((inp[external] == d['bg']).all()), 'frame has extra external contact')
    cargo_labels = np.full(inp.shape, -1, np.int16)
    for i, o in enumerate(d['objects']):
        for p in o['cells']:
            cargo_labels[p] = i
    for o in d['objects']:
        mask = np.zeros(inp.shape, bool)
        for p in o['cells']:
            mask[p] = True
        neighbors = cargo_labels[_halo(mask) & ~mask]
        _check(bool((neighbors < 0).all()), 'cargo components touch')
    for gap, spur in d['repairs']:
        _check(all(not d['wire_input'][p] and clean[p] for p in gap), 'bad gap')
        _check(all(d['wire_input'][p] and not clean[p] for p in spur), 'bad deviation')
    _check(not np.array_equal(inp, out), 'identity example')
    return d


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    n, t = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= n <= 10 and 1 <= t <= 4 and n > t, 'pair counts')
    seen, infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _analyze_pair(pair)
            g = np.asarray(pair['input'], dtype=np.int8)
            key = g.shape, g.tobytes()
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            if split == 'train':
                infos.append(d)
    records = [(f['color'], n, a) for d in infos for f, n, a in zip(d['frames'], d['counts'], d['areas'])]
    _check(any(c == cc and a == aa and n != nn for c, n, a in records for cc, nn, aa in records),
           'same-color equal-area different-count witness missing')
    _check(any(n == nn and a != aa for c, n, a in records for cc, nn, aa in records),
           'area independence missing')
    _check({0, 1, 2} <= {n for _, n, _ in records}, 'zero/one/two evidence missing')
    _check(any(d['filled_counts'] != d['counts'] for d in infos), 'intentional-gap contrast missing')
    _check(any(d['global_counts'] != d['counts'] for d in infos), 'network ownership contrast missing')
    _check(any(any(f['color'] == ff['color'] and n != nn
                         for f, n in zip(d['frames'], d['counts'])
                         for ff, nn in zip(d['frames'], d['counts'])) for d in infos),
           'repeated-category unequal-local-count witness missing')
    _check(any(any(o['color'] not in {f['color'] for f in d['frames']} and bool(ids)
                       for o, ids in zip(d['objects'], d['touched'])) for d in infos),
           'attached unmatched-color witness missing')
    _check(any(any(_contacts(o['cells'], d['labels'])[1] > 1
                       and any(o['color'] == f['color']
                               and int(d['labels'][f['port']]) in ids for f in d['frames'])
                       for o, ids in zip(d['objects'], d['touched'])) for d in infos),
           'multiple contact edges must still mean one object')


def _signature(puzzle):
    digest = hashlib.sha256()
    for s in ('train', 'test'):
        for p in puzzle[s]:
            for k in ('input', 'output'):
                a = np.asarray(p[k], np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 fixed seeds, keeping only bounded aggregate measurements."""
    signatures, shapes, masks, palettes, frames, counts, directions = (set() for _ in range(7))
    cycles = repair_total = pairs = 0
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze_pair(pair)
                _check(np.array_equal(solve(pair['input']), pair['output']), 'solve disagreement')
                shapes.add(pair['input'].shape)
                palettes.add((d['bg'], d['line']))
                frames.add(len(d['frames']))
                counts.update(d['counts'])
                directions.update(f['direction'] for f in d['frames'])
                repair_total += len(d['repairs'])
                pairs += 1
                for cells in d['groups']:
                    cs = set(cells)
                    edges = sum((r+1, c) in cs for r, c in cs)+sum((r, c+1) in cs for r, c in cs)
                    cycles += edges >= len(cells)
                for o in d['objects']:
                    r0, _, c0, _ = o['box']
                    normalized = tuple((r-r0, c-c0) for r, c in o['cells'])
                    masks.add(normalized)
    _check(len(signatures) == 42, 'repeated whole puzzle')
    _check(len(shapes) >= 30 and len(palettes) >= 35, 'insufficient geometry/palette diversity')
    _check(frames == {1, 2, 3} and {0, 1, 2, 3, 4} <= counts, 'count/network diversity')
    _check(directions == set(STEPS), 'port directions missing')
    _check(len(masks) >= 40 and cycles > 0, 'shape/topology diversity')
    # Solve is equivariant to all rotations/reflections and arbitrary color roles.
    fixture = generate_puzzle(0, 2, 1)
    p = fixture['train'][0]
    for k in range(4):
        for flip in (False, True):
            a, b = np.rot90(p['input'], k), np.rot90(p['output'], k)
            if flip:
                a, b = a[:, ::-1], b[:, ::-1]
            _check(np.array_equal(solve(a), b), 'orientation equivariance')
    perm = np.random.default_rng(17).permutation(10)
    _check(np.array_equal(solve(perm[p['input']]), perm[p['output']]), 'palette equivariance')
    _check(_signature(fixture) == _signature(generate_puzzle(0, 2, 1)), 'nondeterminism')
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            candidate = generate_puzzle(47116864, n, t)
            _check(len(candidate['train']) == n and len(candidate['test']) == t, 'explicit counts')
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Three semantic negative regressions, independently mutating output pixels.
    d = _inspect(p['input'])
    gap = d['repairs'][0][0]
    f = next(f for f, n in zip(d['frames'], d['counts']) if n)
    dr, dc = f['direction']
    dot = f['port'][0]-3*dr, f['port'][1]-3*dc
    cargo = d['objects'][0]['cells'][0]
    for cell, value in ((gap[0], d['bg']), (dot, d['bg']), (cargo, int(p['input'][cargo]))):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in fixture[s]] for s in fixture}
        bad['train'][0]['output'][cell] = value
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad repair/tally/erasure accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes), background_wire_palettes=len(palettes),
                cargo_masks=len(masks), repaired_gaps=repair_total,
                looped_components=cycles, terminal_counts=sorted(frames), census_values=sorted(counts))
