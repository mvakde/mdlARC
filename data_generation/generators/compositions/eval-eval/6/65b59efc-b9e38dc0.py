"""Perforated exposure: identity-coded material controls fine light lanes.

Standard library + NumPy only. All solving data are visible in the input.
"""
from __future__ import annotations

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _expand(a, n):
    return np.repeat(np.repeat(a, n, axis=0), n, axis=1)


def _connected(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    todo = [remaining.pop()]
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while todo:
        y, x = todo.pop()
        for dy, dx in offsets:
            q = y + dy, x + dx
            if q in remaining:
                remaining.remove(q)
                todo.append(q)
    return not remaining


def _fit_flank(values, sign):
    """Fit the longest terminal stair with at least two observed steps.

    At the mouth the phase is a remainder, making the next boundary exact.
    The chamber before this suffix is used literally, never extrapolated back.
    """
    ys = np.asarray(values, dtype=int)
    xs = np.arange(len(ys)) - (len(ys) - 1)
    candidates = []
    for period in (1, 2):
        for phase in range(period):
            predicted = int(ys[-1]) + sign * ((xs + phase) // period)
            length = 0
            for agrees in (ys == predicted)[::-1]:
                if not agrees:
                    break
                length += 1
            if length >= 3 and len(set(ys[-length:].tolist())) >= 3:
                candidates.append((length, period, phase))
    if not candidates:
        raise ValueError("flank lacks two readable outward steps")
    longest = max(c[0] for c in candidates)
    finalists = [c for c in candidates if c[0] == longest]
    if len(finalists) != 1:
        raise ValueError("ambiguous terminal boundary continuation")
    length, period, phase = finalists[0]
    return period, phase, int(ys[-1]), length


def _scene(board, bg, wall, payload, labels):
    """Recover the unique direction, real chamber, and virtual cone."""
    matches = []
    for turn in range(4):
        a = np.rot90(board, -turn)
        points = np.argwhere(a == wall)
        if len(points) < 9:
            continue
        rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
        cap = np.flatnonzero(a[:, rear] == wall)
        if (len(cap) < 3 or np.any(np.diff(cap) != 1)
                or mouth - rear < 3 or mouth >= a.shape[1] - 2):
            continue
        upper, lower = [], []
        valid = True
        for x in range(rear + 1, mouth + 1):
            rows = np.flatnonzero(a[:, x] == wall)
            if len(rows) != 2 or rows[1] - rows[0] < 2:
                valid = False
                break
            upper.append(int(rows[0]))
            lower.append(int(rows[1]))
        if not valid:
            continue
        if cap[0] != upper[0] or cap[-1] != lower[0]:
            continue
        if any(abs(d) > 1 for v in (upper, lower) for d in np.diff(v)):
            continue
        if not _connected(a == wall, diagonal=True):
            continue
        try:
            fits = (_fit_flank(upper, -1), _fit_flank(lower, 1))
        except ValueError:
            continue
        footprint = np.zeros(a.shape, dtype=bool)
        bounds = []
        for x in range(rear + 1, a.shape[1]):
            limits = []
            for values, fit, sign in zip((upper, lower), fits, (-1, 1)):
                period, phase, last, _ = fit
                limit = (values[x - rear - 1] if x <= mouth else
                         last + sign * ((x - mouth + phase) // period))
                limits.append(limit)
            if limits[1] - limits[0] < 2:
                valid = False
                break
            bounds.append(tuple(limits))
            footprint[:, x] = ((np.arange(a.shape[0]) > limits[0]) &
                               (np.arange(a.shape[0]) < limits[1]))
        if not valid or np.any(footprint & (a == wall)):
            continue
        chamber = footprint.copy()
        chamber[:, mouth + 1:] = False
        patch = a == payload
        if (not _connected(chamber) or not _connected(patch)
                or np.any(patch & ~chamber) or patch.sum() > 3):
            continue
        last_payload = int(np.argwhere(patch)[:, 1].max())
        internal = np.isin(a, labels) & chamber
        if np.any(internal) and np.argwhere(internal)[:, 1].min() <= last_payload:
            continue
        matches.append(dict(turn=turn, canonical=a, rear=rear, mouth=mouth,
                            fits=fits, bounds=tuple(bounds),
                            footprint=footprint, chamber=chamber))
    if len(matches) != 1:
        raise ValueError(f"expected one emitter interpretation, found {len(matches)}")
    return matches[0]


def _decode(input_grid):
    a = _grid(input_grid)
    dividers = [(r, int(a[r, 0])) for r in range(a.shape[0])
                if np.all(a[r] == a[r, 0])]
    candidates = []
    for d, wall in dividers:
        n = d - 3
        if n not in (2, 3, 4) or d + 4 >= a.shape[0]:
            continue
        bg = int(a[d - 1, 0])
        if wall == bg:
            continue
        try:
            masks, targets, order = {}, {}, []
            c = 0
            while c + n + 2 <= a.shape[1] and a[0, c] == wall:
                card = a[:n + 2, c:c + n + 2]
                if (not np.all(card[0] == wall) or not np.all(card[-1] == wall)
                        or not np.all(card[:, 0] == wall)
                        or not np.all(card[:, -1] == wall)):
                    raise ValueError("bad library frame")
                motif = card[1:-1, 1:-1]
                colors = np.unique(motif[motif != bg])
                if len(colors) != 1:
                    raise ValueError("each template needs one visible source identity")
                label = int(colors[0])
                if label == wall or label in masks:
                    raise ValueError("ambiguous source identity")
                target = int(a[n + 2, c + 1 + n // 2])
                if target == bg:
                    raise ValueError("key swatch must be visible")
                order.append(label)
                masks[label] = motif == label
                targets[label] = target
                c += n + 3
            k = len(order)
            if not 2 <= k <= 30 // (n + 3):
                raise ValueError("bad library count")
            # Reconstruct the metadata exactly; no arbitrary marks are ignored.
            header = np.full((d, a.shape[1]), bg, dtype=np.int8)
            for i, label in enumerate(order):
                start = i * (n + 3)
                header[:n + 2, start:start + n + 2] = wall
                header[1:n + 1, start + 1:start + n + 1] = np.where(masks[label], label, bg)
                header[n + 2, start + 1 + n // 2] = targets[label]
            if not np.array_equal(a[:d], header):
                raise ValueError("malformed library or legend")
            top = d + 1
            ends = np.flatnonzero(a[top] == wall)
            if (len(ends) < 5 or ends[0] != 0 or np.any(np.diff(ends) != 1)
                    or ends[-1] == a.shape[1] - 1):
                raise ValueError("bad scene frame top")
            right = int(ends[-1])
            end_rows = np.flatnonzero(a[top:, 0] == wall)
            if not len(end_rows) or np.any(np.diff(end_rows) != 1):
                raise ValueError("bad scene frame side")
            bottom = top + int(end_rows[-1])
            frame = a[top:bottom + 1, :right + 1]
            if (not np.all(frame[-1] == wall) or not np.all(frame[:, -1] == wall)
                    or np.any(a[top:, right + 1:] != bg)
                    or np.any(a[bottom + 1:] != bg)):
                raise ValueError("bad frame or padding")
            board = frame[1:-1, 1:-1].copy()
            if min(board.shape) < 6 or max(board.shape) * n > 30:
                raise ValueError("scene cannot fit the expanded answer")
            colors = set(map(int, np.unique(board)))
            unknown = colors - {bg, wall, *order}
            if len(unknown) != 1 or not set(order) <= colors:
                raise ValueError("scene must identify payload and use every template")
            payload = unknown.pop()
            geometry = _scene(board, bg, wall, payload, order)
            candidates.append(dict(n=n, k=k, bg=bg, wall=wall, payload=payload,
                                   order=tuple(order), masks=masks, targets=targets,
                                   board=board, geometry=geometry,
                                   padding=(a.shape[1] - right - 1,
                                            a.shape[0] - bottom - 1)))
        except (ValueError, IndexError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"expected one encoded scene, found {len(candidates)}")
    return candidates[0]


def _materialize(p):
    """Expand physical material in page coordinates, with source colors."""
    board, n = p['board'], p['n']
    base = _expand(board, n)
    material = np.zeros(base.shape, dtype=bool)
    for y, x in np.argwhere(np.isin(board, p['order'])):
        label = int(board[y, x])
        tile = np.s_[y*n:(y+1)*n, x*n:(x+1)*n]
        base[tile] = np.where(p['masks'][label], label, p['bg'])
        material[tile] = p['masks'][label]
    return base, material


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only lane scan. No seed, examples, or generation state."""
    p = _decode(input_grid)
    base, material = _materialize(p)
    turn = p['geometry']['turn']
    a = np.rot90(base, -turn)
    opaque = np.rot90(material, -turn)
    footprint = _expand(p['geometry']['footprint'], p['n'])
    out = a.copy()
    for y in range(a.shape[0]):
        blocked = False
        for x in range(a.shape[1]):
            if not footprint[y, x]:
                continue
            if opaque[y, x]:
                if not blocked:
                    out[y, x] = p['targets'][int(a[y, x])]
                blocked = True
            elif a[y, x] == p['bg'] and not blocked:
                out[y, x] = p['payload']
    return np.rot90(out, turn).copy()


def _exposure(p):
    """Construction/check masks from a prefix union, separate from solve's scan."""
    base, material = _materialize(p)
    turn = p['geometry']['turn']
    a = np.rot90(base, -turn)
    opaque = np.rot90(material, -turn)
    footprint = _expand(p['geometry']['footprint'], p['n'])
    hit = opaque & footprint
    prefix = np.logical_or.accumulate(hit, axis=1)
    prior = np.zeros_like(prefix)
    prior[:, 1:] = prefix[:, :-1]
    exposed = hit & ~prior
    hidden = hit & prior
    lit_air = footprint & (a == p['bg']) & ~prior
    shadow_air = footprint & (a == p['bg']) & prior
    result = a.copy()
    result[lit_air] = p['payload']
    for label in p['order']:
        result[exposed & (a == label)] = p['targets'][label]
    return dict(base=a, opaque=opaque, footprint=footprint, exposed=exposed,
                hidden=hidden, lit_air=lit_air, shadow_air=shadow_air,
                output=np.rot90(result, turn).copy())


def _features(p):
    e, g = _exposure(p), p['geometry']
    n = p['n']
    command = np.isin(g['canonical'], p['order']) & g['footprint']
    coarse_prior = np.zeros(command.shape, dtype=bool)
    coarse_prior[:, 1:] = np.logical_or.accumulate(command, axis=1)[:, :-1]
    prior_tiles = _expand(coarse_prior, n)
    page_exposed = np.rot90(e['exposed'], g['turn'])
    page_hidden = np.rot90(e['hidden'], g['turn'])
    partial = 0
    for y, x in np.argwhere(np.isin(p['board'], p['order'])):
        tile = np.s_[y*n:(y+1)*n, x*n:(x+1)*n]
        if np.any(page_exposed[tile]) and np.any(page_hidden[tile]):
            partial += 1
    keyed_ids = [label for label in p['order'] if p['targets'][label] != label
                 and np.any(e['exposed'] & (e['base'] == label))]
    masks = tuple(_key(p['masks'][label]) for label in p['order'])
    wallpoints = np.argwhere(g['canonical'] == p['wall'])
    outline = g['canonical'][wallpoints[:, 0].min():wallpoints[:, 0].max()+1,
                             g['rear']:g['mouth']+1] == p['wall']
    widths = [hi-lo for lo, hi in g['bounds'][:g['mouth']-g['rear']]]
    normalized = np.full(p['board'].shape, -1, dtype=np.int8)
    normalized[p['board'] == p['wall']] = -2
    normalized[p['board'] == p['payload']] = -3
    for i, label in enumerate(p['order']):
        normalized[p['board'] == label] = i
    return dict(n=n, k=p['k'], turn=g['turn'], shape=p['board'].shape,
                palette=(p['bg'], p['wall'], p['payload']),
                periods=tuple(f[0] for f in g['fits']),
                phases=tuple(f[1] for f in g['fits']),
                projection=g['canonical'].shape[1]-g['mouth']-1,
                outline=_key(outline), layout=_key(normalized), motifs=masks,
                nonmonotone=bool(np.any(np.diff(widths) < 0)),
                asymmetric=bool(len({sum(b) for b in g['bounds'][:g['mouth']-g['rear']]}) > 1),
                commands=int(np.isin(p['board'], p['order']).sum()),
                internal=int((np.isin(g['canonical'], p['order']) & g['chamber']).sum()),
                outside=int((np.isin(g['canonical'], p['order']) & ~g['footprint']).sum()),
                keyed_ids=len(keyed_ids), partial=partial,
                lit_air=int(e['lit_air'].sum()), shadow_air=int(e['shadow_air'].sum()),
                hidden=int(e['hidden'].sum()),
                outside_material=int((e['opaque'] & ~e['footprint']).sum()),
                transmitted_air=int((e['lit_air'] & prior_tiles).sum()),
                transmitted_exposure=int((e['exposed'] & prior_tiles).sum()),
                shared_target=len(set(p['targets'].values())) < p['k'],
                duplicate_motif=len(set(masks)) < p['k'],
                clipped=bool(g['footprint'][0].any() or g['footprint'][-1].any()))


def _ablation_differences(p):
    """Count changed pixels for named, plausible simpler rules."""
    e = _exposure(p)
    turn = p['geometry']['turn']
    expected = np.rot90(e['output'], -turn)
    wrong = {}
    no_key = expected.copy()
    no_key[e['exposed']] = e['base'][e['exposed']]
    wrong['no_key'] = no_key
    all_key = expected.copy()
    no_shadow = e['base'].copy()
    no_shadow[e['footprint'] & ~e['opaque'] & (e['base'] == p['bg'])] = p['payload']
    for label in p['order']:
        all_key[e['opaque'] & (e['base'] == label)] = p['targets'][label]
        no_shadow[e['opaque'] & e['footprint'] & (e['base'] == label)] = p['targets'][label]
    wrong['unconditional_key'] = all_key
    wrong['no_shadow'] = no_shadow
    solid = dict(p, masks={c: np.ones_like(m) for c, m in p['masks'].items()})
    wrong['no_template'] = np.rot90(_exposure(solid)['output'], -turn)
    no_cone_geometry = dict(p['geometry'])
    full = np.zeros_like(p['geometry']['footprint'])
    full[:, p['geometry']['rear']+1:] = True
    full[p['geometry']['canonical'] == p['wall']] = False
    no_cone_geometry['footprint'] = full
    no_cone = dict(p, geometry=no_cone_geometry)
    wrong['no_cone'] = np.rot90(_exposure(no_cone)['output'], -turn)
    shifted = p['order'][1:] + p['order'][:1]
    confused = dict(p, masks={a: p['masks'][b] for a, b in zip(p['order'], shifted)},
                    targets={a: p['targets'][b] for a, b in zip(p['order'], shifted)})
    wrong['wrong_identity'] = np.rot90(_exposure(confused)['output'], -turn)
    return {name: int(np.sum(expected != grid)) for name, grid in wrong.items()}


def _motif(rng, n):
    rr, cc = np.indices((n, n))
    style = int(rng.integers(9))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (rr == 0) | (rr == n-1) | (cc == 0) | (cc == n-1)
    elif style == 2:
        mask = rr == cc
    elif style == 3:
        mask = (rr + cc) % 2 == int(rng.integers(2))
    elif style == 4:
        mask = rr == int(rng.integers(n))
    elif style == 5:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif style == 6:
        mask = np.zeros((n, n), dtype=bool)
        y0, y1 = sorted(map(int, rng.integers(n, size=2)))
        x0, x1 = sorted(map(int, rng.integers(n, size=2)))
        mask[y0:y1+1, x0:x1+1] = True
    else:
        mask = rng.random((n, n)) < rng.uniform(.18, .82)
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _geometry(rng, n, projection_mode):
    limit = 30 // n
    for _ in range(128):
        periods = [1, 1] if limit <= 7 else [int(rng.integers(1, 3)) for _ in range(2)]
        phases = [int(rng.integers(v)) for v in periods]
        flare = 2 * max(periods)
        max_trunk = min(5, limit - flare - 3)
        if max_trunk < 1:
            continue
        trunk = int(rng.integers(1, max_trunk + 1))
        profiles = []
        for period, phase in zip(periods, phases):
            neck = 1 if limit <= 7 else int(rng.integers(1, 3))
            body = [neck]
            # An independently shaped reservoir may bulge or pinch.
            for _j in range(trunk - 1):
                body.insert(0, max(1, min(4, body[0] + int(rng.integers(-1, 2)))))
            profile = body + [neck + (t + phase) // period for t in range(1, flare + 1)]
            profiles.append(profile)
        span = max(profiles[0]) + max(profiles[1]) + 1
        axial = trunk + flare
        min_projection = 4 if projection_mode == 'long' else 2
        max_projection = 3 if projection_mode == 'short' else limit - axial - 1
        if span > limit or max_projection < min_projection:
            continue
        projection = int(rng.integers(min_projection, max_projection + 1))
        max_rear = limit - axial - projection - 1
        if max_rear < 0:
            continue
        rear = int(rng.integers(0, min(3, max_rear) + 1))
        width = rear + axial + projection + 1
        height = int(rng.integers(max(7, span), limit + 1))
        center = int(rng.integers(max(profiles[0]), height - max(profiles[1])))
        wall = np.zeros((height, width), dtype=bool)
        footprint = np.zeros_like(wall)
        upper = [center-v for v in profiles[0]]
        lower = [center+v for v in profiles[1]]
        wall[upper[0]:lower[0]+1, rear] = True
        mouth = rear + axial
        for j, (lo, hi) in enumerate(zip(upper, lower), rear + 1):
            wall[lo, j] = wall[hi, j] = True
            footprint[lo+1:hi, j] = True
        # Compute terminal remainders directly from construction parameters.
        for x in range(mouth + 1, width):
            limits = []
            for profile, period, phase, sign in zip(profiles, periods, phases, (-1, 1)):
                extension = ((flare + x-mouth + phase) // period
                             - (flare + phase) // period)
                limits.append(center + sign * (profile[-1] + extension))
            footprint[max(0, limits[0]+1):min(height, limits[1]), x] = True
        return wall, footprint, rear, mouth, center
    raise RuntimeError("bounded chamber construction exhausted")


def _encode(board, n, order, masks, targets, bg, wall, rng):
    library_width = len(order) * (n + 3) - 1
    base_width = max(library_width, board.shape[1] + 2)
    spare = 30 - base_width
    padding = int(rng.integers(1, min(3, spare) + 1))
    bottom_padding = int(rng.integers(3))
    d = n + 3
    h = n + board.shape[0] + 6
    a = np.full((h + bottom_padding, base_width + padding), bg, dtype=np.int8)
    for i, label in enumerate(order):
        c = i * (n + 3)
        a[:n+2, c:c+n+2] = wall
        a[1:n+1, c+1:c+n+1] = np.where(masks[label], label, bg)
        a[n+2, c+1+n//2] = targets[label]
    a[d] = wall
    a[d+1:d+board.shape[0]+3, :board.shape[1]+2] = wall
    a[d+2:d+board.shape[0]+2, 1:board.shape[1]+1] = board
    return a


def _build_pair(rng, n, k, turn, core_roles, mode, teaching, dense):
    palette = list(map(int, rng.permutation(10)))
    if core_roles is None:
        bg, wall, payload = palette[:3]
    else:
        bg, wall, payload = core_roles
    labels = [c for c in palette if c not in (bg, wall, payload)][:k]
    masks = {label: _motif(rng, n) for label in labels}
    if not teaching and k > 2 and rng.random() < .2:
        masks[labels[-1]] = masks[labels[0]].copy()
    targets = {label: int(rng.choice([c for c in range(10) if c != bg])) for label in labels}
    if teaching:
        options = [c for c in range(10) if c not in (bg, *labels[:2])]
        keyed = list(map(int, rng.choice(options, 2, replace=False)))
        targets[labels[0]], targets[labels[1]] = keyed
    elif rng.random() < .25:
        targets[labels[-1]] = targets[labels[0]]
    walls, footprint, rear, mouth, center = _geometry(rng, n, mode)
    board = np.full(walls.shape, bg, dtype=np.int8)
    board[walls] = wall
    board[center, rear + 1] = payload
    # Sometimes use a longer or transverse payload patch, entirely upstream.
    if not teaching and rng.random() < .45:
        candidates = [(y, x) for y, x in np.argwhere(footprint & (board == bg))
                      if x <= rear + 2 and abs(y-center)+abs(x-rear-1) == 1]
        if candidates:
            y, x = candidates[int(rng.integers(len(candidates)))]
            board[y, x] = payload
    patch_end = int(np.argwhere(board == payload)[:, 1].max())
    front = mouth + 1
    back = int(rng.integers(front + 1, board.shape[1]))
    # A controlled pair of serial pieces makes transmission observable.
    board[center, front] = labels[0]
    board[center, back] = labels[1]
    if teaching == 'simple':
        board[center, back] = bg
        options = [(int(y), int(x)) for y, x in np.argwhere(footprint & (board == bg))
                   if y != center and x >= mouth + 1]
        if not options:
            return None
        y, x = options[int(rng.integers(len(options)))]
        board[y, x] = labels[1]
    # Every template is used, and at least one exterior piece stays unexposed.
    for index, label in enumerate(labels[2:]):
        candidates = [(int(y), int(x)) for y, x in np.argwhere(board == bg)
                      if (not footprint[y, x] or x > patch_end)
                      and not (teaching and y == center and x < front and footprint[y, x])]
        if not candidates:
            return None
        y, x = candidates[int(rng.integers(len(candidates)))]
        board[y, x] = label
    exterior = np.argwhere(~footprint & (board == bg))
    if not len(exterior):
        return None
    y, x = exterior[int(rng.integers(len(exterior)))]
    board[y, x] = labels[int(rng.integers(k))]
    extra_count = int(rng.integers(10, 15) if dense == 'heavy' else
                      rng.integers(3, 10) if dense else rng.integers(0, 5))
    for _ in range(extra_count):
        candidates = [(int(y), int(x)) for y, x in np.argwhere(board == bg)
                      if (not footprint[y, x] or x > patch_end)
                      and not (teaching and y == center and x < front and footprint[y, x])]
        if not candidates:
            break
        # Alternate scattered placements with groups in a common coarse row.
        if rng.random() < .35:
            grouped = [q for q in candidates if q[0] == center and q[1] >= front]
            if grouped:
                candidates = grouped
        y, x = candidates[int(rng.integers(len(candidates)))]
        board[y, x] = labels[int(rng.integers(k))]
    page_board = np.rot90(board, turn).copy()
    order = list(map(int, rng.permutation(labels)))
    inp = _encode(page_board, n, order, masks, targets, bg, wall, rng)
    # Construction receives the explicit footprint. The solver later fits rails.
    p = dict(n=n, k=k, bg=bg, wall=wall, payload=payload, masks=masks,
             targets=targets, order=tuple(order), board=page_board,
             geometry=dict(turn=turn, footprint=footprint))
    output = _exposure(p)['output']
    try:
        decoded = _decode(inp)
    except ValueError:
        return None
    if not np.array_equal(output, solve(inp)):
        raise AssertionError("construction and recovered boundary disagree")
    f = _features(decoded)
    if teaching and (f['keyed_ids'] < 2 or f['outside_material'] == 0
                     or f['shadow_air'] == 0 or f['lit_air'] == 0):
        return None
    if teaching == 'serial' and (f['transmitted_air'] == 0 or
                                 f['transmitted_exposure'] == 0 or
                                 f['partial'] == 0 or f['hidden'] == 0):
        return None
    if teaching == 'serial' and not all(_ablation_differences(decoded).values()):
        return None
    return {'input': inp, 'output': output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    base_n = int(rng.choice([2, 3])) if regime != 3 else int(rng.choice([2, 3, 4]))
    base_k = int(rng.integers(2, min(4, 30 // (base_n + 3)) + 1))
    preferred = int(rng.integers(4))
    shared_direction = rng.random() < .25
    core_roles = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .25 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            n, k = base_n, base_k
            mode = 'any'
            teaching = ('simple' if index == 0 else 'serial') if split == 'train' and index < 2 else None
            if regime == 0:
                mode = 'short' if split == 'train' else 'long'
            elif regime == 1:
                k = int(rng.integers(2, 4)) if split == 'train' else int(rng.integers(4, 30 // (n+3)+1))
            elif regime == 2:
                n = (2 if index == 0 else 3 if index == 1 else int(rng.choice([2, 3]))) if split == 'train' else 4
                k = min(k, 30 // (n + 3))
            elif regime == 3 and index >= 2:
                n = int(rng.choice([2, 3, 4]))
                k = int(rng.integers(2, 30 // (n + 3) + 1))
            turn = preferred if shared_direction else int(rng.integers(4))
            # Test difficulty comes from recombination; no new physics is added.
            dense = ('heavy' if split == 'test' and regime == 3 else
                     split == 'test' and regime == 1)
            for _attempt in range(160):
                pair = _build_pair(rng, n, k, turn, core_roles, mode, teaching, dense)
                if pair is None or _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f"bounded unique exposure construction exhausted: {seed}/{split}/{index}")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    ablations = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            p = _decode(inp)
            assert out.shape == tuple(s*p['n'] for s in p['board'].shape)
            e = _exposure(p)
            assert np.array_equal(out, e['output']), "incorrect exposure output"
            assert np.array_equal(out, solve(inp)), "input-only solver mismatch"
            # All physical structure and unexposed material are retained.
            canonical_out = np.rot90(out, -p['geometry']['turn'])
            preserved = ~e['lit_air'] & ~e['exposed']
            assert np.array_equal(canonical_out[preserved], e['base'][preserved])
            assert np.all(canonical_out[e['shadow_air']] == p['bg'])
            assert np.all(canonical_out[e['lit_air']] == p['payload'])
            for label in p['order']:
                selection = e['exposed'] & (e['base'] == label)
                assert np.all(canonical_out[selection] == p['targets'][label])
            assert not np.array_equal(inp, out)
            features[split].append(_features(p))
            if split == 'train':
                ablations.append(_ablation_differences(p))
    train = features['train']
    assert any(f['keyed_ids'] >= 2 and f['outside_material'] > 0
               and f['shadow_air'] > 0 and f['lit_air'] > 0 for f in train), "missing core evidence"
    assert any(f['transmitted_air'] > 0 and f['transmitted_exposure'] > 0
               and f['partial'] > 0 and f['hidden'] > 0 and f['keyed_ids'] >= 2
               for f in train), "missing fine transmission / first-surface evidence"
    assert all(any(a[name] > 0 for a in ablations) for name in ablations[0]), "shortcut not refuted"
    test = features['test']
    assert (min(f['projection'] for f in test) > max(f['projection'] for f in train)
            or min(f['k'] for f in test) > max(f['k'] for f in train)
            or any(f['n'] not in {g['n'] for g in train} for f in test)
            or min(f['commands'] for f in test) > max(f['commands'] for f in train)), "missing train/test shift"
    return None


def _hand_regression():
    """A manually specified scene with a known two-lane serial interaction."""
    board = np.zeros((7, 9), dtype=np.int8)
    board[2:5, 0] = 1
    for x, lo in enumerate((2, 2, 1, 0), 1):
        board[lo, x] = board[6-lo, x] = 1
    board[3, 1] = 2
    board[3, 5] = 3
    board[3, 7] = 4
    board[0, 0] = 4
    masks = {3: np.array([[1, 1], [0, 0]], dtype=bool),
             4: np.array([[1, 0], [1, 1]], dtype=bool)}
    rng = np.random.default_rng(9876)
    inp = _encode(board, 2, [4, 3], masks, {3: 6, 4: 7}, 0, 1, rng)
    out = solve(inp)
    # First lane: the front surface exposes, then even its rear pixel stays 3.
    assert out[6, 10] == 6 and out[6, 11] == 3
    assert out[6, 12] == 0 and out[6, 14] == 4
    # Second lane passes the first piece's empty row, exposing the second piece.
    assert out[7, 10] == 2 and out[7, 12] == 2
    assert out[7, 14] == 7 and out[7, 15] == 4 and out[7, 16] == 0
    # Exterior material and the virtual boundary are unilluminated.
    assert out[0, 0] == 4 and out[0, 1] == 0
    assert out[0, 6] == 0 and out[4, 6] == 2
    # Page-oriented templates stay fixed when the field turns. Supply rotated
    # motifs here so this physical rotation has the corresponding rotated answer.
    for turn in (1, 2, 3):
        rotated_input = _encode(np.rot90(board, turn), 2, [3, 4],
                                {c: np.rot90(m, turn) for c, m in masks.items()},
                                {3: 6, 4: 7}, 0, 1, rng)
        assert np.array_equal(solve(rotated_input), np.rot90(out, turn))


def self_test():
    _hand_regression()
    shapes, outlines, layouts, motifs, palettes, directions, scales, libraries = (set() for _ in range(8))
    all_features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        local_layouts = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                f = _features(p)
                all_features.append(f)
                shapes.add((pair['input'].shape, pair['output'].shape))
                outlines.add(f['outline'])
                layouts.add(f['layout'])
                local_layouts.add(f['layout'])
                motifs.update(f['motifs'])
                palettes.add(f['palette'])
                directions.add(f['turn'])
                scales.add(f['n'])
                libraries.add(f['k'])
        assert len(local_layouts) >= 3
        if seed in (0, 21, 41):
            repeated = generate_puzzle(seed)
            for split in ('train', 'test'):
                for a, b in zip(puzzle[split], repeated[split]):
                    assert all(np.array_equal(a[k], b[k]) for k in ('input', 'output'))
    assert len(signatures) == 42
    assert directions == {0, 1, 2, 3} and scales == {2, 3, 4}
    assert libraries >= {2, 3, 4, 5, 6}
    assert len(shapes) >= 50 and len(outlines) >= 35 and len(layouts) >= 150
    assert len(motifs) >= 75 and len(palettes) >= 50
    for flag in ('nonmonotone', 'asymmetric', 'internal', 'clipped', 'shared_target', 'duplicate_motif'):
        assert sum(bool(f[flag]) for f in all_features) >= 3, flag
    # Host-style explicit counts, including all 30 legal combinations.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(575188821, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bad_seed in (-1, True, 1.5):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Wrong labels and erased material identities must fail the validator.
    valid = generate_puzzle(99, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in valid[s]] for s in valid}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("wrong output accepted")
    p = _decode(valid['test'][0]['input'])
    bad_input = valid['test'][0]['input'].copy()
    n = p['n']
    bad_input[1:n+1, 1:n+1] = p['bg']
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("unidentified template accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'shape_pairs': len(shapes), 'chamber_outlines': len(outlines),
            'placement_layouts': len(layouts), 'template_bitmaps': len(motifs),
            'core_palettes': len(palettes)}
