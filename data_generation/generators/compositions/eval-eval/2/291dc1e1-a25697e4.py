"""Mirror-latch loom: directed ribbon reading with a geometric parity register.

Only NumPy and the standard library are used. No parent modules, files, seed
metadata, or demonstration labels are consulted by the input-only reader.
"""
from itertools import combinations
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, and train > test")


def _grid(value):
    a = np.asarray(value)
    if not (a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
            and np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))):
        raise ValueError("Grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, found = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    found.add(q)
        result.append(frozenset(found))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rect(box):
    r0, r1, c0, c1 = box
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    """Four rotations, then four reflected rotations; normalize jointly."""
    def point(r, c):
        if op >= 4:
            c = -c
        for _ in range(op % 4):
            r, c = c, -r
        return r, c
    raw = [frozenset(point(r, c) for r, c in region) for region in regions]
    r0, _, c0, _ = _bbox(frozenset().union(*raw))
    return tuple(_shift(region, -r0, -c0) for region in raw)


def _shape_signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted(_shift(cells, -r0, -c0)))


def _orbit(cells):
    return {_shape_signature(_transform((cells,), op)[0]) for op in range(8)}


def _canonical(cells):
    return min(_orbit(cells))


def _border(inp):
    """The doubled corner marker marks the fast arm, visibly in every input."""
    matches = []
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                chart = inp[::-1] if bottom else inp
                chart = chart[:, ::-1] if right else chart
                chart = chart.T if axis else chart
                if min(chart.shape) < 5:
                    continue
                origin, fast, slow, bg = map(int, (chart[0, 0], chart[0, 2], chart[1, 0], chart[-1, -1]))
                if (len({origin, fast, slow, bg}) == 4 and chart[0, 1] == origin
                        and np.all(chart[0, 2:] == fast) and np.all(chart[1:, 0] == slow)
                        and np.all(chart[-1, 1:] == bg) and np.all(chart[1:, -1] == bg)):
                    matches.append((chart[1:, 1:].copy(), axis, bottom, right, origin, fast, slow, bg))
    if len(matches) != 1:
        raise ValueError("Input must have one marked L guide and clear opposite margins")
    return matches[0]


def _fits(chart, bg, key_regions, rectangle, slot):
    """Return all legal whole-key fits, considering either color as the plug."""
    sr0, _, sc0, _ = _bbox(slot)
    fits = []
    for op in range(8):
        transformed = _transform(key_regions, op)
        for plug_index in (0, 1):
            pr0, _, pc0, _ = _bbox(transformed[plug_index])
            absolute = tuple(_shift(region, sr0 - pr0, sc0 - pc0) for region in transformed)
            if absolute[plug_index] != slot:
                continue
            handle = absolute[1 - plug_index]
            if handle & rectangle:
                continue
            if any(not (0 <= r < chart.shape[0] and 0 <= c < chart.shape[1])
                   or chart[r, c] != bg for r, c in handle):
                continue
            fits.append((op, plug_index, absolute))
    return fits


def _analyze(input_grid):
    inp = _grid(input_grid)
    chart, axis, bottom, right, origin, fast, slow, bg = _border(inp)
    parts = _components((int(r), int(c)) for r, c in np.argwhere(chart != bg))
    keys = []
    for index, cells in enumerate(parts):
        colors = sorted({int(chart[p]) for p in cells})
        if len(colors) == 2 and len(cells) != len(_rect(_bbox(cells))):
            regions = tuple(frozenset(p for p in cells if chart[p] == color) for color in colors)
            if all(len(_components(region)) == 1 for region in regions):
                keys.append((index, tuple(colors), regions))
    if len(keys) != 1:
        raise ValueError("There must be exactly one nonrectangular two-color calibration key")
    key_index, inks, regions = keys[0]
    strips, mono = [], []
    for index, cells in enumerate(parts):
        if index == key_index:
            continue
        colors = {int(chart[p]) for p in cells}
        box = _bbox(cells)
        if colors <= set(inks) and cells == _rect(box) and box[1] - box[0] == 1 and box[3] > box[2]:
            strips.append({'box': box, 'cells': cells})
        elif len(colors) == 1 and not colors & set(inks):
            mono.append((cells, next(iter(colors))))
        else:
            raise ValueError("Unexpected object: expected ribbons or monochrome lock fragments")
    if not (1 <= len(strips) <= 15 and len(mono) == 2 * len(strips)
            and len({color for cells, color in mono}) == 1):
        raise ValueError("Each ribbon needs two fragments of a common lock color")
    frame = mono[0][1]
    used = set()
    for strip in strips:
        rr0, rr1, cc0, cc1 = strip['box']
        candidates = []
        for i, j in combinations(range(len(mono)), 2):
            cells = mono[i][0] | mono[j][0]
            box = _bbox(cells)
            r0, r1, c0, c1 = box
            if not (r1 == rr0 - 2 and c0 == cc0 and c1 <= cc1 and r0 < r1 and c0 < c1):
                continue
            rectangle = _rect(box)
            if frozenset(p for p in rectangle if chart[p] != bg) != cells:
                continue
            slot = rectangle - cells
            if slot and len(_components(slot)) == 1:
                candidates.append((i, j, box, rectangle, slot))
        if len(candidates) != 1:
            raise ValueError("Lock-to-ribbon association must be unique by alignment and one-row gap")
        i, j, box, rectangle, slot = candidates[0]
        if i in used or j in used:
            raise ValueError("Lock fragments reused between stations")
        used.update((i, j))
        fits = _fits(chart, bg, regions, rectangle, slot)
        parities = {int(op >= 4) for op, plug, absolute in fits}
        if len(parities) != 1:
            raise ValueError("A lock must determine one reflection parity from complete-key fitting")
        strip.update(lock_box=box, rectangle=rectangle, slot=slot, fits=fits, bit=parities.pop())
        strip['block'] = chart[rr0:rr1 + 1, cc0:cc1 + 1].copy()
    if len(used) != len(mono):
        raise ValueError("Unassigned lock fragment")
    strips.sort(key=lambda s: (s['box'][0], s['box'][2]))
    lanes = sorted({s['box'][0] for s in strips})
    if any(b - a < 3 for a, b in zip(lanes, lanes[1:])):
        raise ValueError("Ribbon lane bands overlap or are staggered")
    for lane in lanes:
        intervals = [(s['box'][2], s['box'][3]) for s in strips if s['box'][0] == lane]
        if any(b[0] - a[1] < 2 for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Ribbons in a lane need background separation")
    lengths = [s['block'].shape[1] for s in strips]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("Ribbon lengths must share parity for exact centering")
    reverse_thickness = (bottom != right) if axis == 0 else (bottom == right)
    return dict(chart=chart, configuration=(axis, bottom, right), bg=bg, origin=origin,
                fast=fast, slow=slow, frame=frame, inks=inks, key_regions=regions,
                key_cells=parts[key_index], strips=strips, lanes=lanes, lengths=lengths,
                reverse_thickness=reverse_thickness)


def _render(d, alternative=None):
    strips = d['strips']
    if alternative == 'reverse_fast':
        strips = [s for lane in d['lanes'] for s in reversed([s for s in strips if s['box'][0] == lane])]
    elif alternative == 'reverse_slow':
        strips = [s for lane in reversed(d['lanes']) for s in strips if s['box'][0] == lane]
    elif alternative == 'fast_major':
        strips = sorted(strips, key=lambda s: (s['box'][2], s['box'][0]))
    width = max(d['lengths'])
    out = np.full((2 * len(strips), width), d['bg'], dtype=np.int8)
    state, last_lane = 0, None
    for index, strip in enumerate(strips):
        if alternative == 'lane_reset' and strip['box'][0] != last_lane:
            state = 0
        last_lane = strip['box'][0]
        if alternative != 'delayed':
            state ^= strip['bit']
        polarity = strip['bit'] if alternative == 'local' else state
        if alternative == 'no_swap':
            polarity = 0
        block = strip['block'].copy()
        if d['reverse_thickness'] and alternative != 'no_rotation':
            block = block[::-1]
        if polarity:
            block = np.where(block == d['inks'][0], d['inks'][1], d['inks'][0]).astype(np.int8)
        offset = 0 if alternative == 'left_align' else (width - block.shape[1]) // 2
        out[2 * index:2 * index + 2, offset:offset + block.shape[1]] = block
        if alternative == 'delayed':
            state ^= strip['bit']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fit locks, accumulate their reflection bits, normalize and stack ribbons."""
    return _render(_analyze(input_grid))


def _sample_shape(rng, fallback=False):
    if fallback:
        slot = frozenset({(1, 0), (1, 1), (2, 1), (2, 2), (2, 3)})
        handle = frozenset({(1, -1), (0, -1)})
        return _rect((0, 3, 0, 3)) - slot, slot, handle
    for _ in range(100):
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
        if rng.random() < .24:
            # Symmetric L slots: the other key region can resolve parity.
            h = int(rng.integers(3, 5))
            w = int(rng.integers(2, h + 1))
            row = int(rng.integers(1, h - 1))
            slot = {(row, c) for c in range(w)} | {(r, 0) for r in range(row, h)}
        else:
            row = int(rng.integers(1, h - 1))
            slot = {(row, 0)}
            for col in range(1, w):
                new = int(rng.integers(1, h - 1))
                slot.update((r, col) for r in range(min(row, new), max(row, new) + 1))
                row = new
            if rng.random() < .35:
                frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIR4
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - slot)
                if frontier:
                    slot.add(frontier[int(rng.integers(len(frontier)))])
        rectangle = _rect((0, h - 1, 0, w - 1))
        target = rectangle - slot
        if len(_components(target)) != 2 or _bbox(target) != (0, h - 1, 0, w - 1):
            continue
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIR4
                         if (r + dr, c + dc) not in rectangle})
        handle = {starts[int(rng.integers(len(starts)))]}
        size = int(rng.integers(2, 8))
        style = int(rng.integers(3))
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIR4
                               if -2 <= r + dr < h + 2 and -2 <= c + dc < w + 2}
                              - rectangle - handle)
            if not frontier:
                break
            degrees = np.array([sum((r + dr, c + dc) in handle for dr, dc in DIR4)
                                for r, c in frontier], dtype=float)
            weights = degrees ** 2 if style == 0 else 1 / degrees if style == 1 else np.ones(len(frontier))
            handle.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
        union = frozenset(slot | handle)
        if len(handle) == size and union != _rect(_bbox(union)):
            return frozenset(target), frozenset(slot), frozenset(handle)
    return _sample_shape(rng, True)


def _station(shape, op, padding, length):
    target, plug, handle = shape
    r0, r1, c0, c1 = _bbox(target)
    top, bottom = padding
    target = _rect((r0 - top, r1 + bottom, c0, c1)) - plug
    target, plug, handle = _transform((target, plug, handle), op)
    r0, r1, c0, c1 = _bbox(target)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    # Relative ribbon rows are 0 and 1. Its lock ends at row -2.
    dr, dc = -r1 - 2, -c0
    target, plug, handle = (_shift(region, dr, dc) for region in (target, plug, handle))
    if w > length or any(r in (0, 1) and c >= 0 for r, c in handle):
        raise ValueError("Assembly does not fit its ribbon station")
    ribbon = _rect((0, 1, 0, length - 1))
    footprint = target | plug | handle | ribbon
    return dict(target=target, plug=plug, handle=handle, footprint=footprint,
                box=_bbox(footprint), length=length, op=op, cap_shape=(h, w))


def _block(rng, length, inks, witness=False):
    if witness:
        for _ in range(100):
            b = rng.choice(inks, size=(2, length)).astype(np.int8)
            if all(not np.array_equal(b, other) for other in (b[::-1], b[:, ::-1], b[::-1, ::-1])):
                return b
        raise AssertionError("Could not produce asymmetric binary ribbon")
    mode = int(rng.integers(5))
    if mode == 0:
        return np.full((2, length), rng.choice(inks), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(inks, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(inks, size=(2, 1)), length, axis=1).astype(np.int8)
    if mode == 3:
        motif = rng.choice(inks, size=(2, int(rng.integers(1, min(4, length) + 1))))
        return np.tile(motif, (1, length))[:, :length].astype(np.int8)
    return rng.choice(inks, size=(2, length)).astype(np.int8)


def _physical(chart, configuration, roles):
    bg, origin, fast, slow, frame, ink_a, ink_b = roles
    g = np.full((chart.shape[0] + 1, chart.shape[1] + 1), bg, dtype=np.int8)
    g[1:, 1:] = chart
    g[0, 0:2] = origin
    g[0, 2:] = fast
    g[1:, 0] = slow
    axis, bottom, right = configuration
    if axis:
        g = g.T
    if bottom:
        g = g[::-1]
    if right:
        g = g[:, ::-1]
    return g.copy()


def _sample_roles(rng):
    bg, origin, fast, slow = map(int, rng.choice(10, size=4, replace=False))
    if rng.random() < .45:
        pool = [c for c in range(10) if c not in (bg, origin, fast, slow)]
    else:
        # Guides are identified on the border; field colors may reuse them.
        pool = [c for c in range(10) if c != bg]
    frame, ink_a, ink_b = map(int, rng.choice(pool, size=3, replace=False))
    return bg, origin, fast, slow, frame, ink_a, ink_b


def _build_once(rng, configuration, roles, bits, lane_counts, witness, fallback=False):
    shape = _sample_shape(rng, fallback)
    key_op = int(rng.choice((0, 2, 4, 6))) if fallback and len(lane_counts) == 3 else int(rng.integers(8))
    key_regions = _transform(shape[1:], key_op)
    key_box = _bbox(key_regions[0] | key_regions[1])
    key_h, key_w = key_box[1] + 1, key_box[3] + 1
    parity = int(rng.integers(2))
    min_length = max(_bbox(shape[0])[1] + 1, _bbox(shape[0])[3] + 1) + 1
    min_length += (parity - min_length) % 2
    lengths = [min_length + 2 * int(rng.integers(4)) for _ in bits]
    if witness:
        lengths[0], lengths[-1] = min_length, min_length + 2
    stations = []
    for index, bit in enumerate(bits):
        options = list(range(4 * (bit ^ int(key_op >= 4)), 4 * (bit ^ int(key_op >= 4)) + 4))
        rng.shuffle(options)
        if fallback and len(lane_counts) == 3:
            # These two orientations need an extra upstream row. Prefer the
            # equally valid compact fits for a three-lane canvas.
            options.sort(key=lambda op: op in (1, 7))
        for op in options:
            try:
                # Padding changes rectangle context without changing the plug.
                padding = (int(rng.integers(2)), int(rng.integers(2))) if not fallback else (0, 0)
                station = _station(shape, op, padding, lengths[index])
                stations.append(station)
                break
            except ValueError:
                continue
        else:
            raise ValueError("No station orientation with the requested parity fits")
    groups, start = [], 0
    for count in lane_counts:
        group = stations[start:start + count]
        start += count
        # Narrow overfull lanes rather than imposing the longest first.
        while sum(s['box'][3] - s['box'][2] + 1 for s in group) + count + 1 > 29:
            eligible = [i for i, s in enumerate(group) if s['length'] > min_length]
            if not eligible:
                raise ValueError("Too many stations across the fast axis")
            chosen = eligible[int(rng.integers(len(eligible)))]
            old = group[chosen]
            # Reducing only the ribbon preserves the lock and hypothetical handle.
            length = old['length'] - 2
            footprint = old['target'] | old['plug'] | old['handle'] | _rect((0, 1, 0, length - 1))
            old.update(length=length, footprint=footprint, box=_bbox(footprint))
        groups.append(group)
    lane_extents = [(-min(s['box'][0] for s in group), max(s['box'][1] for s in group)) for group in groups]
    slow_min = key_h + 2 + sum(a + b + 1 for a, b in lane_extents) + len(groups)
    fast_min = max(key_w + 2, max(sum(s['box'][3] - s['box'][2] + 1 for s in group) + len(group) + 1
                                  for group in groups))
    if slow_min > 29 or fast_min > 29:
        raise ValueError("Objects do not fit within the ARC canvas")
    slow_extent = slow_min + int(rng.integers(min(4, 29 - slow_min) + 1))
    fast_extent = fast_min + int(rng.integers(min(5, 29 - fast_min) + 1))
    bg, origin, fast, slow, frame, ink_a, ink_b = roles
    chart = np.full((slow_extent, fast_extent), bg, dtype=np.int8)
    shelf_near = bool(rng.integers(2))
    shelf_top = 1 if shelf_near else slow_extent - key_h - 1
    key_col = int(rng.integers(1, fast_extent - key_w))
    for region, color in zip(key_regions, (ink_a, ink_b)):
        for p in _shift(region, shelf_top, key_col):
            chart[p] = color
    cursor = key_h + 2 if shelf_near else 1
    # Distribute spare slow-axis space between lanes and the margins.
    spare = slow_extent - slow_min
    slow_extras = rng.multinomial(spare, np.ones(len(groups) + 1) / (len(groups) + 1))
    cursor += int(slow_extras[0])
    blocks, all_lengths = [], []
    axis, bottom, right = configuration
    reverse_thickness = (bottom != right) if axis == 0 else (bottom == right)
    reserved = set()
    for lane_index, (group, (above, below)) in enumerate(zip(groups, lane_extents)):
        ribbon_row = cursor + above
        widths = [s['box'][3] - s['box'][2] + 1 for s in group]
        extra = fast_extent - sum(widths) - len(group) - 1
        gaps = 1 + rng.multinomial(extra, np.ones(len(group) + 1) / (len(group) + 1))
        col_cursor = int(gaps[0])
        for station, width, gap_after in zip(group, widths, gaps[1:]):
            ribbon_col = col_cursor - station['box'][2]
            absolute = _shift(station['footprint'], ribbon_row, ribbon_col)
            if absolute & reserved or any(chart[p] != bg for p in absolute):
                raise ValueError("Station collides with calibration key or another station")
            reserved.update(absolute)
            for p in _shift(station['target'], ribbon_row, ribbon_col):
                chart[p] = frame
            block = _block(rng, station['length'], (ink_a, ink_b), witness)
            if not witness and blocks and station['length'] == blocks[-1].shape[1] and rng.random() < .25:
                block = blocks[-1].copy()
            physical = block[::-1] if reverse_thickness else block
            chart[ribbon_row:ribbon_row + 2, ribbon_col:ribbon_col + station['length']] = physical
            blocks.append(block)
            all_lengths.append(station['length'])
            col_cursor += width + int(gap_after)
        cursor = ribbon_row + below + 2 + int(slow_extras[lane_index + 1])
    if witness and len(set(all_lengths)) == 1:
        raise ValueError("Witness should show nonzero centering padding")
    inp = _physical(chart, configuration, roles)
    # Construct labels from sampled sequence and bits, separately from fitting.
    out = np.full((2 * len(blocks), max(all_lengths)), bg, dtype=np.int8)
    state = 0
    for index, (block, bit) in enumerate(zip(blocks, bits)):
        state ^= bit
        printed = np.where(block == ink_a, ink_b, ink_a).astype(np.int8) if state else block
        left = (out.shape[1] - block.shape[1]) // 2
        out[2 * index:2 * index + 2, left:left + block.shape[1]] = printed
    d = _analyze(inp)
    if [s['bit'] for s in d['strips']] != bits or not np.array_equal(_render(d), out):
        raise ValueError("Visible fitting does not recover the intended gate sequence")
    return {'input': inp, 'output': out}


def _build_pair(rng, configuration, roles, bits, lane_counts, witness):
    for attempt in range(100):
        try:
            return _build_once(rng, configuration, roles, bits, lane_counts, witness, fallback=attempt >= 70)
        except ValueError:
            continue
    raise RuntimeError("Bounded station construction exhausted")


ALTERNATIVES = ('no_swap', 'local', 'delayed', 'lane_reset', 'reverse_fast',
                'reverse_slow', 'fast_major', 'no_rotation', 'left_align')


def _training_evidence(descriptors):
    if {d['configuration'][0] for d in descriptors} != {0, 1}:
        raise ValueError("Demonstrations must show both fast-axis orientations")
    for alternative in ALTERNATIVES:
        if not any(not np.array_equal(_render(d), _render(d, alternative)) for d in descriptors):
            raise ValueError("Demonstrations fail to distinguish " + alternative)
    if not any(s['bit'] == 0 for d in descriptors for s in d['strips']):
        raise ValueError("No rotation witness")
    if not any(s['bit'] == 1 for d in descriptors for s in d['strips']):
        raise ValueError("No reflection witness")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    configurations = [(axis, bottom, right) for axis in (0, 1)
                      for bottom in (False, True) for right in (False, True)]
    first = configurations[int(rng.integers(8))]
    second = (1 - first[0], not first[1], not first[2])
    held = [c for c in configurations if c not in (first, second)]
    stable_palette = bool(rng.random() < .3)
    palette = _sample_roles(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            witness = split == 'train' and index < 2
            if witness:
                configuration = first if index == 0 else second
                bits = [0, 1, 0, 1] if index == 0 else [1, 0, 1, 0]
                lane_counts = [2, 2] if index == 0 else [1, 3]
            else:
                pool = (first, second) if split == 'train' else held
                configuration = pool[int(rng.integers(len(pool)))]
                n = int(rng.integers(1, 7) if split == 'train' else rng.integers(3, 9))
                # Mixed, all-rotation, and all-reflection station sequences.
                mode = int(rng.integers(4))
                bits = [mode] * n if mode < 2 else list(map(int, rng.integers(2, size=n)))
                if n == 1:
                    lane_counts = [1]
                elif n <= 3:
                    lane_counts = [n] if rng.random() < .5 else [1, n - 1]
                elif n >= 7 or (n >= 5 and rng.random() < .2):
                    lane_counts = [1, 1, 1]
                    for _ in range(n - 3):
                        eligible = [j for j, count in enumerate(lane_counts) if count < 3]
                        lane_counts[eligible[int(rng.integers(len(eligible)))]] += 1
                else:
                    cut = int(rng.integers(max(1, n - 3), min(3, n - 1) + 1))
                    lane_counts = [cut, n - cut]
            for _ in range(10):
                roles = palette if stable_palette else _sample_roles(rng)
                pair = _build_pair(rng, configuration, roles, bits, lane_counts, witness)
                signature = _signature(pair['input'])
                if signature not in seen:
                    if split == 'train' and index == 1:
                        try:
                            _training_evidence([_analyze(p['input']) for p in puzzle['train']] + [_analyze(pair['input'])])
                        except ValueError:
                            continue
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not create distinct, sufficient demonstrations")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("Puzzle needs exactly train/test keys")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("Splits must be lists")
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("Pairs need exactly input/output keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            if signature in seen:
                raise ValueError("Duplicate input")
            seen.add(signature)
            d = _analyze(inp)
            expected = _render(d)
            if not np.array_equal(expected, out):
                raise ValueError("Incorrect cumulative parity, ribbon orientation, order, or centering")
            if out.shape != (2 * len(d['strips']), max(d['lengths'])):
                raise ValueError("Incorrect output dimensions")
            # Gate/key/frame/guide cells are interpreted, never copied to output.
            if not set(map(int, out.flat)) <= {d['bg'], *d['inks']}:
                raise ValueError("Output contains non-ink role colors")
            (training if split == 'train' else testing).append(d)
    _training_evidence(training)
    train_configs = {d['configuration'] for d in training}
    if any(d['configuration'] in train_configs for d in testing):
        raise ValueError("Generated tests must use a new combination of demonstrated guide factors")


def puzzle_features(puzzle):
    """Measurements read from returned inputs, not generation-time parameters."""
    rows = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            d = _analyze(pair['input'])
            bits = [s['bit'] for s in d['strips']]
            widths = d['lengths']
            used_ops = sorted({op for s in d['strips'] for op, plug, absolute in s['fits']})
            rows.append(dict(split=split, index=index, input_shape=list(np.asarray(pair['input']).shape),
                             output_shape=list(np.asarray(pair['output']).shape), configuration=d['configuration'],
                             roles=(d['bg'], d['origin'], d['fast'], d['slow'], d['frame'], *d['inks']),
                             station_count=len(bits), lane_counts=[sum(s['box'][0] == lane for s in d['strips'])
                                                                  for lane in d['lanes']],
                             bits=bits, lengths=widths, operations=used_ops,
                             first_max=widths.index(max(widths)), max_multiplicity=widths.count(max(widths)),
                             key_shapes=[_canonical(r) for r in d['key_regions']],
                             slot_orbits=[len(_orbit(s['slot'])) for s in d['strips']],
                             cap_shapes=[(s['lock_box'][1] - s['lock_box'][0] + 1,
                                          s['lock_box'][3] - s['lock_box'][2] + 1) for s in d['strips']],
                             plug_handle_sizes=[(len(d['key_regions'][s['fits'][0][1]]),
                                                 len(d['key_regions'][1 - s['fits'][0][1]])) for s in d['strips']],
                             carry_witness=any(sum(bits[:i]) % 2 and bit == 0 for i, bit in enumerate(bits)),
                             repeated_ribbons=len({_signature(s['block']) for s in d['strips']}) < len(bits)))
    return rows


def _maximum_count_fixture():
    """Fifteen locks on three five-station lanes, with reflection at every lock."""
    roles = (0, 9, 1, 2, 5, 3, 4)
    plug = frozenset({(1, 0), (1, 1), (1, 2), (2, 2)})
    handle = frozenset({(1, -1), (0, -1)})
    target = _rect((0, 2, 0, 2)) - plug
    chart = np.zeros((26, 26), dtype=np.int8)
    # The template is mirrored relative to every lock, making all fifteen bits 1.
    for region, color in zip(_transform((plug, handle), 4), roles[-2:]):
        for point in _shift(region, 1, 1):
            chart[point] = color
    block = np.array([[3, 3, 4], [4, 3, 4]], dtype=np.int8)
    for lane in range(3):
        for column in range(5):
            top, left = 5 + 7 * lane, 2 + 5 * column
            for point in _shift(target, top, left):
                chart[point] = roles[4]
            chart[top + 4:top + 6, left:left + 3] = block
    expected = np.concatenate([7 - block if i % 2 == 0 else block for i in range(15)], axis=0)
    return _physical(chart, (0, False, False), roles), expected


def self_test():
    """Bounded 42-seed variation suite, count checks, and semantic negatives."""
    rows, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        rows.extend(puzzle_features(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
        # Inputs have a visible guide, so rotating the whole scene is equivariant.
        pair = puzzle['train'][0]
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(pair['input'], rotation)), pair['output'])
        mirrored_expected = pair['output'].reshape(-1, 2, pair['output'].shape[1])[:, ::-1, :].reshape(pair['output'].shape)
        assert np.array_equal(solve(pair['input'][:, ::-1]), mirrored_expected)
    assert len(signatures) == 42
    assert {tuple(r['configuration']) for r in rows} == {(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)}
    assert len({tuple(r['input_shape']) for r in rows}) >= 30
    assert len({tuple(r['output_shape']) for r in rows}) >= 10
    assert len({tuple(r['roles']) for r in rows}) >= 40
    assert len({tuple(map(tuple, shape)) for r in rows for shape in r['key_shapes']}) >= 30
    assert {op for r in rows for op in r['operations']} == set(range(8))
    assert {r['station_count'] for r in rows} == set(range(1, 9))
    assert {len(r['lane_counts']) for r in rows} == {1, 2, 3}
    assert {int(np.sign(a - b)) for r in rows for a, b in r['plug_handle_sizes']} == {-1, 0, 1}
    assert any(min(r['slot_orbits']) < 8 for r in rows)
    assert any(r['repeated_ribbons'] for r in rows)
    assert sum(r['first_max'] > 0 for r in rows) >= 40
    assert any(len(set(r['lengths'])) == 1 for r in rows)
    maximum_input, maximum_output = _maximum_count_fixture()
    assert maximum_output.shape == (30, 3)
    maximum = _analyze(maximum_input)
    assert len(maximum['strips']) == 15 and len(maximum['lanes']) == 3
    assert [s['bit'] for s in maximum['strips']] == [1] * 15
    assert np.array_equal(solve(maximum_input), maximum_output)
    assert not np.array_equal(_render(maximum, 'lane_reset'), maximum_output)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(991 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(1001, 3, 1), generate_puzzle(1001, 3, 1)
    assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
               for x, y in zip(p[s], q[s]) for k in ('input', 'output'))
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("Damaged output accepted")
    # Explicit two-color ambiguous bar: fitting rotations and reflections would
    # produce different register states, so never choose an arbitrary symmetry.
    chart = np.zeros((12, 12), dtype=np.int8)
    rectangle = _rect((4, 6, 2, 4))
    slot = _rect((5, 5, 2, 4))
    for point in rectangle - slot:
        chart[point] = 5
    regions = (frozenset({(0, 0), (0, 1), (0, 2)}), frozenset({(0, 3), (1, 3)}))
    assert {int(op >= 4) for op, pi, absolute in _fits(chart, 0, regions, rectangle, slot)} == {0, 1}
    return dict(seeds=42, pairs=len(rows), distinct_puzzles=len(signatures),
                input_shapes=len({tuple(r['input_shape']) for r in rows}),
                output_shapes=len({tuple(r['output_shape']) for r in rows}),
                key_region_shapes=len({tuple(map(tuple, shape)) for r in rows for shape in r['key_shapes']}),
                symmetric_slot_pairs=sum(min(r['slot_orbits']) < 8 for r in rows),
                repeated_ribbon_pairs=sum(r['repeated_ribbons'] for r in rows),
                nonfirst_max_pairs=sum(r['first_max'] > 0 for r in rows))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
