"""Travel-clock loom: leading enclosed cavities project a travel-shifted ribbon.

Only NumPy and the standard library are used. Labels are drawn from explicit
bar intervals and declared cavity rows; solve reconstructs everything from pixels.
"""
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC colors")
    return a.astype(int, copy=False)


def _components(mask):
    """Four-connected components, in raster order."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = deque([(r, c)]), []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _corner_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _annotations(a):
    candidates = []
    for raw in np.unique(a):
        pos = np.argwhere(a == raw)
        if len(pos) != 6:
            continue
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        if min(r1 - r0, c1 - c0) < 1:
            continue
        bounds = r0, r1, c0, c1
        if set(map(tuple, pos)) != _corner_cells(bounds):
            continue
        local = a[r0:r1 + 1, c0:c1 + 1]
        if sorted(map(len, _components(local == raw))) != [3, 3]:
            continue
        other = np.unique(local[local != raw])
        if len(other) == 1:
            candidates.append((int(raw), int(other[0]), bounds))
    _require(len(candidates) == 1, "expected one pair of corner L annotations")
    return candidates[0]


def _runs(indices):
    indices = list(map(int, indices))
    groups = []
    for x in indices:
        if not groups or x != groups[-1][-1] + 1:
            groups.append([x])
        else:
            groups[-1].append(x)
    return groups


def _arithmetic(values, name):
    _require(len(values) >= 3, f"not enough observed {name}")
    delta = values[1] - values[0]
    _require(delta >= 0 and all(v == values[0] + i * delta for i, v in enumerate(values)),
             f"{name} must have nonnegative arithmetic growth")
    return int(values[0]), int(delta)


def _read_ribbon(line, bg, rail):
    caps = np.flatnonzero(line == rail)
    _require(len(caps) == 1, "one observation endcap required")
    end = int(caps[0])
    _require(end >= 7 and np.all(line[end + 1:] == bg), "malformed observation endcap")
    groups = _runs(np.flatnonzero(line[:end] != bg))
    _require(4 <= len(groups) <= 5 and groups[0][0] == 0 and groups[-1][-1] == end - 1,
             "ribbon must expose four or five complete bars starting beside rail")
    colors = []
    for group in groups:
        vals = np.unique(line[group])
        _require(len(vals) == 1 and vals[0] not in (bg, rail), "bars must be monochrome")
        colors.append(int(vals[0]))
    widths = [len(g) for g in groups]
    gaps = [b[0] - a[-1] - 1 for a, b in zip(groups, groups[1:])]
    w0, dw = _arithmetic(widths, "bar widths")
    q0, dq = _arithmetic(gaps, "bar gaps")
    periods = [p for p in (2, 3) if len(set(colors[:p])) == p
               and len(colors) >= p + 1
               and all(c == colors[i % p] for i, c in enumerate(colors))]
    _require(len(periods) == 1, "ribbon color cycle is not established")
    p = periods[0]
    return dict(w0=w0, dw=dw, q0=q0, dq=dq, palette=tuple(colors[:p]),
                observed=len(groups), end=end, widths=tuple(widths), gaps=tuple(gaps))


def _ribbon_value(model, tick, bg):
    """Pixel-derived recurrence; no generation state or observed endpoint cutoff."""
    _require(tick >= 0, "negative clock tick")
    start, i = 0, 0
    while True:
        width = model['w0'] + i * model['dw']
        gap = model['q0'] + i * model['dq']
        if tick < start + width:
            return model['palette'][i % len(model['palette'])]
        if tick < start + width + gap:
            return bg
        start += width + gap
        i += 1


def _object_details(work, bg, rail_col):
    objects = []
    previous = 0
    for cells in _components(work != bg):
        rr, cc = np.asarray(cells).T
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        _require(r0 >= 2 and r0 > previous + 1, "objects require separated row bands below ribbon")
        previous = r1
        colors = {int(work[r, c]) for r, c in cells}
        _require(len(colors) == 1, "each connected object must be monochrome")
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rr - r0, cc - c0] = True
        active, decoy, active_rows = [], [], set()
        holes = _holes(mask)
        for hole in holes:
            rows = {r for r, c in hole}
            contacts = {r for r, c in hole if c == mask.shape[1] - 2 and mask[r, -1]}
            if contacts:
                _require(rows == contacts, "partially leading cavities are outside this family")
                active.append(hole)
                active_rows.update(rows)
            else:
                decoy.append(hole)
        enclosed = {p for hole in holes for p in hole}
        naive_rows = ({r for r in range(mask.shape[0]) if not mask[r, -2]}
                      if mask.shape[1] >= 2 else set())
        open_rows = {r for r in naive_rows if (r, mask.shape[1] - 2) not in enclosed}
        objects.append(dict(cells=cells, row=r0, end_row=r1, mask=mask,
                            color=colors.pop(), shift=rail_col - 1 - c1,
                            active=active, decoy=decoy, active_rows=active_rows,
                            naive_rows=naive_rows, open_rows=open_rows))
    _require(objects, "work side has no objects")
    return objects


def _scene(input_grid):
    a = _grid(input_grid)
    marker, bg, bounds = _annotations(a)
    clean = a.copy()
    clean[clean == marker] = bg
    candidates = []
    for turn in range(4):
        canonical = np.rot90(clean, turn)
        h, w = canonical.shape
        for col in range(1, w - 1):
            rail = int(canonical[0, col])
            if rail in (bg, marker) or not np.all(canonical[:, col] == rail):
                continue
            if np.count_nonzero(canonical == rail) != h + 1:
                continue
            if not np.all(canonical[1:, col + 1:] == bg):
                continue
            if not np.all(canonical[:2, :col] == bg):
                continue
            try:
                ribbon = _read_ribbon(canonical[0, col + 1:], bg, rail)
                objects = _object_details(canonical[:, :col], bg, col)
                # The annotation window must be wholly on the projection side,
                # with no calibration row included, in the inferred orientation.
                window = np.zeros(a.shape, dtype=bool)
                r0, r1, c0, c1 = bounds
                window[r0:r1 + 1, c0:c1 + 1] = True
                yy, xx = np.nonzero(np.rot90(window, turn))
                _require(yy.min() >= 2 and xx.min() > col, "crop is outside projection band")
                _require(all(o['color'] not in (bg, marker, rail) for o in objects), "reserved object color")
            except AssertionError:
                continue
            candidates.append(dict(canonical=canonical, turn=turn, background=bg,
                                   marker=marker, bounds=bounds, rail=rail, col=col,
                                   ribbon=ribbon, objects=objects,
                                   crop=(int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max()))))
    _require(len(candidates) == 1, "orientation, rail and ribbon must be unique")
    return candidates[0]


def _prediction(info, *, shift_clock=True, gate='active', repeat_observation=False):
    """The optional modes are explicit ablations used in validation/self_test."""
    a, bg, col = info['canonical'], info['background'], info['col']
    field = np.full(a.shape, bg, dtype=int)
    for obj in info['objects']:
        rows = obj['active_rows'] if gate == 'active' else obj['naive_rows']
        if gate == 'all_holes':
            rows = {r for hole in obj['active'] + obj['decoy'] for r, c in hole}
        for r in rows:
            for c in range(col + 1, a.shape[1]):
                tick = c - col - 1 + (obj['shift'] if shift_clock else 0)
                if repeat_observation:
                    tick %= info['ribbon']['end']
                field[obj['row'] + r, c] = _ribbon_value(info['ribbon'], tick, bg)
    full = np.rot90(field, -info['turn'])
    r0, r1, c0, c1 = info['bounds']
    return full[r0:r1 + 1, c0:c1 + 1].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the rule's parameters and answer using only this input."""
    return _prediction(_scene(input_grid))


def _intervals(rng, height, count):
    spare = height - (2 * count + 1)
    _require(spare >= 0, "too many cavities")
    bins = rng.multinomial(spare, np.full(2 * count + 1, 1 / (2 * count + 1)))
    row, result = 1 + int(bins[0]), []
    for i in range(count):
        length = 1 + int(bins[2 * i + 1])
        result.append((row, row + length))
        row += length + 1 + int(bins[2 * i + 2])
    return result


def _shape(rng, kind, max_height=9, max_width=8):
    """Explicit carving grammar. Return a mask and declared emitting rows."""
    min_h = 5 if kind == 'multi' else 2 if kind == 'solid' else 3
    min_w = 6 if kind == 'mixed' else 4 if kind == 'decoy' else 2 if kind == 'solid' else 3
    h = int(rng.integers(min_h, max(min_h, max_height) + 1))
    w = int(rng.integers(min_w, max(min_w, max_width) + 1))
    mask, active = np.ones((h, w), dtype=bool), set()
    slots = min(4, (h - 1) // 2)

    def carve(left, right, count, emits):
        for start, stop in _intervals(rng, h, count):
            # Row-dependent back edges produce connected, nonrectangular holes.
            back = int(rng.integers(left, right + 1))
            for r in range(start, stop):
                edge = int(rng.integers(left, right + 1)) if rng.random() < .45 else back
                mask[r, edge:right + 1] = False
            if emits:
                active.update(range(start, stop))

    if kind == 'solid':
        if h > 2 and w > 2 and rng.random() < .65:
            spine = int(rng.integers(w))
            for r in range(h):
                mask[r, :int(rng.integers(spine + 1))] = False
                mask[r, int(rng.integers(spine, w)) + 1:] = False
    elif kind == 'open':
        mask[1:-1, 1:-1] = False
        side = int(rng.integers(4))
        if side < 2:
            mask[1:-1, 0 if side == 0 else -1] = False
        else:
            mask[0 if side == 2 else -1, -2] = False
    elif kind == 'recessed':
        extension = int(rng.integers(1, 4))
        enlarged = np.zeros((h, w + extension), dtype=bool)
        enlarged[:, :w] = mask
        enlarged[int(rng.choice([0, h - 1])), w:] = True
        mask = enlarged
        carve(1, w - 2, int(rng.integers(1, slots + 1)), False)
    elif kind == 'decoy':
        carve(1, w - 3, int(rng.integers(1, slots + 1)), False)
    elif kind == 'mixed':
        separator = int(rng.integers(2, w - 2))
        carve(separator + 1, w - 2, int(rng.integers(1, slots + 1)), True)
        carve(1, separator - 1, int(rng.integers(1, slots + 1)), False)
    else:
        count = int(rng.integers(2 if kind == 'multi' else 1, slots + 1))
        carve(1, w - 2, count, True)
        if count >= 2 and rng.random() < .5:
            # Some divider rows become a one-cell spine joining hollow lobes.
            for r in range(1, h - 1):
                if mask[r].all() and mask[r - 1].all() and mask[r + 1].all():
                    mask[r, :-1] = False
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
    mask = mask[r0:r1 + 1, c0:c1 + 1].copy()
    active = {r - r0 for r in active}
    _require(len(_components(mask)) == 1, "shape grammar broke connectivity")
    return mask, active


def _roles(rng):
    colors = tuple(map(int, rng.permutation(10)))
    # Object colors may coincide with ribbon colors; spatial roles identify them.
    return dict(bg=colors[0], rail=colors[1], marker=colors[2],
                palette=colors[3:6], object_colors=colors[3:])


def _sample_clock(rng, level):
    n = int(rng.integers(4, 6))
    p = int(rng.integers(2, 4))
    dw = int(rng.choice([0, 1, 2], p=[.40, .50, .10])) if level else 0
    dq = int(rng.choice([0, 1], p=[.65, .35])) if level else 0
    if dw == 2:
        n = 4
    return dict(w0=int(rng.integers(1, 3)), dw=dw,
                q0=int(rng.integers(1, 3)), dq=dq, observed=n, p=p)


def _bar_intervals(clock, stop):
    """Ground-truth closed-form positions, separate from the solver recurrence."""
    i = 0
    while True:
        start = i * (clock['w0'] + clock['q0']) + i * (i - 1) // 2 * (clock['dw'] + clock['dq'])
        if start > stop:
            return
        width = clock['w0'] + i * clock['dw']
        yield i, start, start + width
        i += 1


def _draw_pair(rng, roles, clock, shapes, distances, turn, crop_mode, mandatory):
    bg, rail, marker = roles['bg'], roles['rail'], roles['marker']
    palette = roles['palette'][:clock['p']]
    observed = list(_bar_intervals(clock, 200))[:clock['observed']]
    end = observed[-1][2]
    work = max(mask.shape[1] + d for (mask, rows), d in zip(shapes, distances))
    # Reserve at least one unobserved tick beyond the observation endcap.
    spare = 30 - work - 1 - (end + 2)
    if spare < 0:
        return None
    col = work + int(rng.integers(min(3, spare) + 1))
    band = int(rng.integers(end + 2, 30 - col))
    base = sum(mask.shape[0] for mask, rows in shapes) + len(shapes) - 1 + 2
    if base > 30:
        return None
    height = base + int(rng.integers(min(5, 30 - base) + 1))
    extras = rng.multinomial(height - base, np.full(len(shapes) + 1, 1 / (len(shapes) + 1)))
    a = np.full((height, col + 1 + band), bg, dtype=int)
    a[:, col] = rail
    for i, start, stop in observed:
        a[0, col + 1 + start:col + 1 + stop] = palette[i % len(palette)]
    a[0, col + 1 + end] = rail
    field = np.full(a.shape, bg, dtype=int)
    order = list(map(int, rng.permutation(len(shapes))))
    row, placed = 2 + int(extras[0]), []
    max_tick = band - 1 + max(distances)
    latent_bars = list(_bar_intervals(clock, max_tick))
    for j, k in enumerate(order):
        mask, active_rows = shapes[k]
        d = distances[k]
        x = col - mask.shape[1] - d
        color = roles['object_colors'][0] if mandatory else int(rng.choice(roles['object_colors']))
        patch = a[row:row + mask.shape[0], x:x + mask.shape[1]]
        patch[mask] = color
        # Shift each latent bar interval back by travel distance, then intersect
        # with the projection band; cavity rows gate this geometric drawing.
        for i, start, stop in latent_bars:
            lo, hi = max(0, start - d), min(band, stop - d)
            if lo < hi:
                for r in active_rows:
                    field[row + r, col + 1 + lo:col + 1 + hi] = palette[i % len(palette)]
        placed.append((row, row + mask.shape[0] - 1))
        row += mask.shape[0] + 1 + int(extras[j + 1])
    if mandatory:
        r0, r1 = placed[0][0], placed[-1][1]
    else:
        r0, r1 = placed[0][0], placed[-1][1]
        if rng.random() < .4 and r1 - r0 >= 4:
            r0 = int(rng.integers(r0, r1 - 1))
        r1 = max(r1, r0 + 2)
        r1 = min(height - 1, r1)
    if r1 - r0 < 2:
        return None
    if crop_mode == 'far' and not mandatory:
        x0 = int(rng.integers(max(1, end // 2), band - 2))
    else:
        x0 = int(rng.integers(min(4, band - 2)))
    x1 = band - 1
    if not mandatory and rng.random() < .45:
        x1 = int(rng.integers(x0 + 2, band))
    bb = (r0, r1, col + 1 + x0, col + 1 + x1)
    mask_window = np.zeros(a.shape, dtype=bool)
    mask_window[r0:r1 + 1, bb[2]:bb[3] + 1] = True
    a, field, mask_window = (np.rot90(v, turn).copy() for v in (a, field, mask_window))
    yy, xx = np.nonzero(mask_window)
    bounds = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
    for r, c in _corner_cells(bounds):
        a[r, c] = marker
    r0, r1, c0, c1 = bounds
    return {'input': a, 'output': field[r0:r1 + 1, c0:c1 + 1].copy()}


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _unique(pair, seen):
    # Twenty color bijections, all distinct on an input with at least four
    # colors, exceed the maximum thirteen earlier inputs. No rule changes.
    for sign in (1, -1):
        for offset in range(10):
            lookup = (sign * np.arange(10) + offset) % 10
            p = {k: lookup[v] for k, v in pair.items()}
            if _key(p['input']) not in seen:
                return p
    raise AssertionError("bounded color orbit exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    level = int(rng.choice([0, 1, 2], p=[.18, .48, .34]))
    shift = str(rng.choice(['distance', 'crop', 'orientation', 'composition']))
    base_turn = int(rng.integers(4))
    varying_roles = level > 0 and rng.random() < .65
    roles = _roles(rng)
    # Bounded clock fitting: the final fallback is a short two-color clock.
    for attempt in range(41):
        clock = (_sample_clock(rng, level) if attempt < 40 else
                 dict(w0=1, dw=1 if level else 0, q0=1, dq=0, observed=4, p=2))
        obs = list(_bar_intervals(clock, 200))[:clock['observed']]
        if obs[-1][2] <= (14 if shift == 'distance' else 18):
            break
    h, w = int(rng.integers(4, 7)), int(rng.integers(3, 6))
    closed = np.ones((h, w), dtype=bool)
    closed[1:-1, 1:-1] = False
    opened = closed.copy()
    side = int(rng.integers(3))
    if side == 0:
        opened[1:-1, 0] = False
    else:
        opened[0 if side == 1 else -1, -2] = False
    extension = int(rng.integers(1, 3))
    recessed = np.zeros((h, w + extension), dtype=bool)
    recessed[:, :w] = closed
    recessed[int(rng.choice([0, h - 1])), w:] = True
    emitting = set(range(1, h - 1))
    anchor_indices = list(map(int, rng.permutation(nt)[:2]))
    requirements = {
        anchor_indices[0]: ([(closed, emitting), (closed.copy(), emitting), (opened, set())], [0, 1, 1]),
        anchor_indices[1]: ([(closed.copy(), emitting), (recessed, set())], [2, 0]),
    }
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            mandatory = split == 'train' and index in requirements
            pair = None
            for attempt in range(81):
                current_roles = roles if mandatory or not varying_roles else _roles(rng)
                current_clock = clock if mandatory or level == 0 else _sample_clock(rng, level)
                turn = base_turn
                if split == 'test' and shift == 'orientation':
                    turn = (base_turn + 1 + index % 3) % 4
                elif level == 2 and not mandatory and shift != 'orientation':
                    turn = int(rng.integers(4))
                if mandatory:
                    shapes, distances = requirements[index]
                else:
                    count_objects = int(rng.integers(1, 4 if level == 0 else 6))
                    if split == 'test' and shift == 'composition':
                        count_objects = int(rng.integers(3, 6))
                    max_h = max(3, (27 - count_objects) // count_objects)
                    silent = rng.random() < .13
                    pool = ['solid', 'open', 'recessed', 'decoy'] if silent else (
                        ['active', 'active', 'open', 'recessed', 'solid'] if level == 0 else
                        ['active', 'multi', 'mixed', 'open', 'recessed', 'decoy', 'solid'])
                    kinds = [str(rng.choice(pool)) for _ in range(count_objects)]
                    if not silent:
                        kinds[0] = 'active'
                    shapes = [_shape(rng, k, min(10, max_h), 5 if shift == 'distance' else 8) for k in kinds]
                    distances = [int(rng.integers(4, 7)) if split == 'test' and shift == 'distance'
                                 else int(rng.integers(0, 4)) for _ in shapes]
                if attempt == 80:
                    # Guaranteed fit, preserving mandatory evidence and the
                    # designated shift. This changes only visible clock geometry.
                    current_clock = dict(w0=1, dw=0, q0=1, dq=0, observed=4, p=2)
                    if not mandatory:
                        if split == 'test' and shift == 'composition':
                            shapes = [(closed.copy(), emitting) for _ in range(3)]
                            distances = [0, 1, 2]
                        else:
                            shapes = [(closed.copy(), emitting)]
                            distances = [4 if split == 'test' and shift == 'distance' else 2]
                candidate = _draw_pair(rng, current_roles, current_clock, shapes, distances,
                                       turn, 'far' if split == 'test' and shift == 'crop' else 'near', mandatory)
                if candidate is None:
                    continue
                try:
                    info = _scene(candidate['input'])
                except AssertionError:
                    # Reject accidental annotation/scene ambiguity in a sampled
                    # silhouette. The final ring construction is unambiguous.
                    continue
                _require(np.array_equal(_prediction(info), candidate['output']), "label/solver disagreement")
                if mandatory:
                    if not np.any(candidate['output'] != info['background']):
                        continue
                    if np.array_equal(candidate['output'], _prediction(info, shift_clock=False)):
                        continue
                    if np.array_equal(candidate['output'], _prediction(info, gate='all_holes')) and index == anchor_indices[1]:
                        continue
                    if np.array_equal(candidate['output'], _prediction(info, gate='naive')) and index == anchor_indices[0]:
                        continue
                pair = _unique(candidate, seen)
                break
            _require(pair is not None, "bounded pair construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _scene(a)
            _require(np.array_equal(b, _prediction(info)), "incorrect output")
            _require(not np.any(b == info['marker']), "annotation leaked into answer")
            _require(set(map(int, np.unique(b))) <= {info['background'], *info['ribbon']['palette']},
                     "output has colors outside the ribbon/background roles")
            # Docking must be a legal rigid translation inside each original
            # row band; whole-object extreme, rather than local hole edge, wins.
            for obj in info['objects']:
                moved = [(r, c + obj['shift']) for r, c in obj['cells']]
                _require(obj['shift'] >= 0 and max(c for r, c in moved) == info['col'] - 1,
                         "object does not dock at the rail")
                _require(len(set(moved)) == int(obj['mask'].sum()), "rigid motion changed object")
            if split == 'train':
                training.append((info, b))
    objects = [o for info, b in training for o in info['objects']]
    _require(any(o['active'] for o in objects), "no active training cavities")
    _require(any(o['open_rows'] for o in objects), "no deep open training contrast")
    _require(any(o['decoy'] and not o['active'] for o in objects), "no recessed training contrast")
    _require(any(o['shift'] == 0 and o['active'] for o in objects), "no already-docked active example")
    _require(any(len(o['active_rows']) >= 2 for o in objects), "no multirow projection evidence")
    for kwargs, message in [({'shift_clock': False}, 'travel has no demonstrated effect'),
                            ({'gate': 'naive'}, 'enclosure has no demonstrated effect'),
                            ({'gate': 'all_holes'}, 'leading extreme has no demonstrated effect'),
                            ({'repeat_observation': True}, 'extrapolation has no demonstrated effect')]:
        _require(any(not np.array_equal(b, _prediction(info, **kwargs)) for info, b in training), message)
    _require(any(len(set(map(int, np.unique(b))) - {info['background']}) >= 2 for info, b in training),
             "training fails to expose multiple ribbon color phases")
    _require(any(info['crop'][3] - info['col'] - 1 + o['shift'] >= info['ribbon']['end']
                 for info, b in training for o in info['objects'] if o['active']),
             "training never probes beyond observed ribbon")


def self_test():
    """Bounded 42-seed audit, including semantic ablations and corruption tests."""
    stats = dict(seeds=42, pairs=0, travel_ablation_failures=0, enclosure_ablation_failures=0,
                 all_holes_ablation_failures=0, repeat_prefix_failures=0, silent_pairs=0,
                 growing_width_pairs=0, growing_gap_pairs=0, multi_active_objects=0,
                 mixed_objects=0, stepped_holes=0)
    # Hand-calculated integration fixture: widths 1,2,3,4,...; colors 2,4;
    # one-cell gaps. F(8..17) = 0444402222. Only a leading closed ring
    # emits; a top-door ring and an otherwise identical tailed ring do not.
    fixture = np.zeros((16, 25), dtype=int)
    fixture[:, 8] = 1
    fixture[0, 9:22] = list(map(int, '2044022204444'))
    fixture[0, 22] = 1
    ring = np.ones((4, 4), dtype=bool)
    ring[1:3, 1:3] = False
    fixture[2:6, 2:6][ring] = 7  # right extreme 5, rail 8: travel two.
    opened = ring.copy()
    opened[0, 2] = False
    fixture[7:11, 2:6][opened] = 7
    tail = np.zeros((4, 5), dtype=bool)
    tail[:, :4] = ring
    tail[0, 4] = True
    fixture[12:16, 2:7][tail] = 7
    bounds = (2, 15, 15, 24)
    for r, c in _corner_cells(bounds):
        fixture[r, c] = 9
    expected = np.zeros((14, 10), dtype=int)
    expected[1] = expected[2] = list(map(int, '0444402222'))
    _require(np.array_equal(solve(fixture), expected), "hand-calculated clock/gate regression failed")
    for kwargs in ({'shift_clock': False}, {'gate': 'naive'}, {'gate': 'all_holes'},
                   {'repeat_observation': True}):
        _require(not np.array_equal(_prediction(_scene(fixture), **kwargs), expected),
                 "integration fixture does not distinguish an ablation")
    signatures, shapes, geometries, directions, widths, gaps, cycles, distances, counts = (set() for _ in range(9))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple((_key(p['input']), _key(p['output']))
                          for s in ('train', 'test') for p in puzzle[s])
        _require(signature == tuple((_key(p['input']), _key(p['output']))
                                     for s in ('train', 'test') for p in again[s]), "nondeterminism")
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info, b = _scene(pair['input']), pair['output']
                stats['pairs'] += 1
                directions.add(info['turn'])
                shapes.add((pair['input'].shape, b.shape))
                widths.add(info['ribbon']['dw'])
                gaps.add(info['ribbon']['dq'])
                cycles.add(len(info['ribbon']['palette']))
                counts.add(len(info['objects']))
                stats['growing_width_pairs'] += info['ribbon']['dw'] > 0
                stats['growing_gap_pairs'] += info['ribbon']['dq'] > 0
                stats['silent_pairs'] += not np.any(b != info['background'])
                for key, kwargs in [('travel_ablation_failures', {'shift_clock': False}),
                                    ('enclosure_ablation_failures', {'gate': 'naive'}),
                                    ('all_holes_ablation_failures', {'gate': 'all_holes'}),
                                    ('repeat_prefix_failures', {'repeat_observation': True})]:
                    stats[key] += not np.array_equal(b, _prediction(info, **kwargs))
                for obj in info['objects']:
                    distances.add(obj['shift'])
                    geometries.add(_key(obj['mask']))
                    stats['multi_active_objects'] += len(obj['active']) >= 2
                    stats['mixed_objects'] += bool(obj['active'] and obj['decoy'])
                    for hole in obj['active'] + obj['decoy']:
                        rr, cc = np.asarray(hole).T
                        stats['stepped_holes'] += int(len(hole) != (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
    _require(len(signatures) == 42 and len(shapes) >= 70 and len(geometries) >= 80, "structural diversity collapsed")
    _require(directions == set(range(4)) and cycles == {2, 3}, "missing orientations or cycles")
    _require({0, 1, 2} <= widths and gaps == {0, 1}, "missing clock progressions")
    _require({0, 1, 2, 4, 5, 6} <= distances and {1, 2, 3, 4, 5} <= counts, "motion/count diversity collapsed")
    _require(stats['growing_width_pairs'] >= 30 and stats['growing_gap_pairs'] >= 15,
             "too little geometric extrapolation")
    _require(stats['repeat_prefix_failures'] >= 50 and stats['multi_active_objects'] >= 10
             and stats['mixed_objects'] >= 5 and stats['stepped_holes'] >= 15,
             "clock/cavity geometry collapsed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Equivariance with canonical Ls redrawn after rotating the full scene.
    p = generate_puzzle(11, 2, 1)['train'][0]
    info = _scene(p['input'])
    clean = p['input'].copy()
    clean[clean == info['marker']] = info['background']
    window = np.zeros(clean.shape, dtype=bool)
    r0, r1, c0, c1 = info['bounds']
    window[r0:r1 + 1, c0:c1 + 1] = True
    for turn in range(4):
        a, mask = np.rot90(clean, turn).copy(), np.rot90(window, turn)
        rr, cc = np.nonzero(mask)
        bb = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        for r, c in _corner_cells(bb):
            a[r, c] = info['marker']
        _require(np.array_equal(solve(a), np.rot90(p['output'], turn)), "rotation failed")
    # A changed test label must fail validation.
    broken = generate_puzzle(902, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    stats.update(input_output_shapes=len(shapes), object_geometries=len(geometries),
                 directions=sorted(directions), width_steps=sorted(widths), gap_steps=sorted(gaps),
                 cycle_lengths=sorted(cycles), travel_distances=sorted(distances), object_counts=sorted(counts))
    return stats
