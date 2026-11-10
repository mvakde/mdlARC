"""Centered stencil printing on a reconstructed four-piece surface.

All interpretation data are pixels in the input. No parent module is imported.
Generation keeps its construction model; solve resegments and searches the input.
"""
from __future__ import annotations

import itertools
import math
from collections import deque

import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts: train 2..10, test 1..4, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must lie in 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for dr, dc in NEIGHBORS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
                    cells.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _crop(mask):
    coords = np.argwhere(mask)
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy(), tuple(map(int, low))


def _turn_point(r, c, shape, turns):
    h, w = shape
    for _ in range(turns % 4):
        r, c, h, w = w - 1 - c, r, w, h
    return int(r), int(c)


def _bitmask(mask, side, offset):
    r0, c0 = offset
    return sum(1 << ((int(r) + r0) * side + int(c) + c0)
               for r, c in np.argwhere(mask))


def _placements(piece, side, corner, fixed=False):
    options = []
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    for k in ((0,) if fixed else range(4)):
        mask = np.rot90(piece['mask'], k)
        h, w = mask.shape
        if h > side or w > side:
            continue
        r0 = 0 if corner < 2 else side - h
        c0 = 0 if corner % 2 == 0 else side - w
        cells = set((int(r) + r0, int(c) + c0) for r, c in np.argwhere(mask))
        if [p in cells for p in corners] != [i == corner for i in range(4)]:
            continue
        commands = []
        frozen = []
        for r, c, label in piece['commands']:
            rr, cc = _turn_point(r, c, piece['mask'].shape, k)
            commands.append((rr + r0, cc + c0, label, k, corner))
            frozen.append((r + r0, c + c0, label, k, corner))
        options.append(dict(mask=_bitmask(mask, side, (r0, c0)), cells=tuple(sorted(cells)),
                            commands=tuple(commands), frozen=tuple(frozen), turns=k))
    return options


def _packings(pieces, side):
    """All permitted corner packings (at most 6*4**3), including symmetries."""
    if sum(int(p['mask'].sum()) for p in pieces) != side * side:
        return []
    anchors = _placements(pieces[0], side, 0, fixed=True)
    if not anchors:
        return []
    table = {(i, c): _placements(pieces[i], side, c)
             for i in range(1, 4) for c in range(1, 4)}
    results = []
    full = (1 << (side * side)) - 1

    def visit(corner, remaining, occupied, selected):
        if corner == 4:
            if occupied == full:
                owner = np.empty((side, side), dtype=np.int8)
                commands, frozen, motions = [], [], []
                for j, (index, option) in enumerate(selected):
                    for r, c in option['cells']:
                        owner[r, c] = j
                    commands.extend(option['commands'])
                    frozen.extend(option['frozen'])
                    motions.append((index, option['turns']))
                results.append(dict(owner=owner, commands=tuple(commands),
                                    frozen=tuple(frozen), motions=tuple(motions)))
            return
        for i in remaining:
            for option in table[(i, corner)]:
                if not occupied & option['mask']:
                    visit(corner + 1, tuple(j for j in remaining if j != i),
                          occupied | option['mask'], selected + [(i, option)])

    visit(1, (1, 2, 3), anchors[0]['mask'], [(0, anchors[0])])
    return results


def _parse(input_grid):
    grid = _grid(input_grid)
    frame, bg = int(grid[0, 0]), int(grid[-1, 0])
    _require(frame != bg, "frame and background must differ")
    height = 0
    while height < len(grid) and grid[height, 0] == frame:
        height += 1
    t = height - 3
    _require(t in (3, 5), "unrecognized stencil header")
    library_width = 4 * t + 5
    _require(grid.shape[1] >= library_width and grid.shape[0] > t + 5, "truncated input")
    header = grid[:t + 2, :library_width]
    expected_frame = np.zeros(header.shape, dtype=bool)
    expected_frame[0] = expected_frame[-1] = True
    expected_frame[:, ::t + 1] = True
    _require(np.all(header[expected_frame] == frame), "broken panel frames")
    _require(np.all(grid[:t + 2, library_width:] == bg), "nonblank header margin")
    _require(np.all(grid[t + 2] == frame), "missing scene divider")
    library, order = {}, []
    for j in range(4):
        col = 1 + j * (t + 1)
        panel = grid[1:t + 1, col:col + t]
        colors = np.unique(panel[panel != bg])
        _require(len(colors) == 1, "a stencil needs one identity")
        label = int(colors[0])
        _require(label != frame and label not in library, "stencil identities must be distinct")
        mask = panel == label
        _require(mask[t // 2, t // 2], "stencil center must be marked")
        library[label] = mask.copy()
        order.append(label)
    scene = grid[t + 3:]
    _require(np.all(scene[0] == bg) and np.all(scene[-1] == bg)
             and np.all(scene[:, 0] == bg) and np.all(scene[:, -1] == bg), "scene needs blank margins")
    _require(not np.any(scene == frame), "frame color cannot be a scene role")
    components = _components(scene != bg)
    _require(len(components) == 4, "exactly four connected pieces required")
    for a, b in itertools.combinations(components, 2):
        _require(np.max(np.abs(a[:, None] - b[None, :]), axis=2).min() >= 2,
                 "pieces need a background-cell gap")
    ids = set(library)
    key_patches = []
    for r in range(scene.shape[0] - 1):
        for c in range(scene.shape[1] - 1):
            if set(map(int, scene[r:r + 2, c:c + 2].flat)) == ids:
                key_patches.append((r, c))
    _require(len(key_patches) == 1, "exactly one four-identity legend required")
    kr, kc = key_patches[0]
    key_cells = {(kr + dr, kc + dc) for dr in (0, 1) for dc in (0, 1)}
    key = scene[kr:kr + 2, kc:kc + 2].copy().reshape(-1)
    anchor, others, used = None, [], set()
    for coords in components:
        cells = set(map(tuple, coords.tolist()))
        has_key = key_cells <= cells
        mask = np.zeros(scene.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        small, (r0, c0) = _crop(mask)
        values = scene[coords[:, 0], coords[:, 1]]
        carriers = set(map(int, values)) - ids
        _require(len(carriers) == 1 and not carriers & {bg, frame}, "one carrier per piece required")
        commands = [(int(r - r0), int(c - c0), int(scene[r, c])) for r, c in coords
                    if int(scene[r, c]) in ids and (int(r), int(c)) not in key_cells]
        _require(commands, "every piece needs a printing command")
        for a, b in itertools.combinations(commands, 2):
            _require(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2, "commands must be isolated")
        if has_key:
            for r, c, _ in commands:
                _require(min(max(abs(r + r0 - rr), abs(c + c0 - cc)) for rr, cc in key_cells) >= 2,
                         "commands must be separated from the legend")
        used.update(x[2] for x in commands)
        piece = dict(mask=small, commands=tuple(commands), carrier=carriers.pop(),
                     source_origin=(r0, c0), key_origin=(kr - r0, kc - c0) if has_key else None)
        if has_key:
            _require(anchor is None, "multiple anchor pieces")
            anchor = piece
        else:
            others.append(piece)
    _require(anchor is not None and used == ids, "missing anchor or unused identity")
    total = sum(len(c) for c in components)
    side = math.isqrt(total)
    _require(side * side == total and 7 <= side <= 10, "footprints must form a supported square")
    return dict(side=side, t=t, background=bg, frame=frame, library=library,
                order=tuple(order), key=key, pieces=[anchor] + others, scene_shape=scene.shape)


def _assembly(parsed):
    packings = _packings(parsed['pieces'], parsed['side'])
    unique = {}
    for p in packings:
        # A symmetric footprint is allowed only when its decorated local frame
        # is equivalent: same centers, identities, and effective stencils.
        stamps = tuple(sorted((r, c, label, corner,
                               np.rot90(parsed['library'][label], k).tobytes())
                              for r, c, label, k, corner in p['commands']))
        signature = (p['owner'].tobytes(), stamps)
        unique.setdefault(signature, p)
    _require(len(unique) == 1, "surface must have one decorated corner assembly")
    return next(iter(unique.values()))


def _coverage(parsed, assembly, mode='correct'):
    side, t = parsed['side'], parsed['t']
    counts = np.zeros((side, side), dtype=int)
    possible_colors = np.zeros((4, side, side), dtype=bool)
    outside = 0
    commands = assembly['frozen'] if mode == 'frozen_commands' else assembly['commands']
    positional = dict(zip(map(int, parsed['key']), parsed['order']))
    for r, c, label, k, corner in commands:
        chosen = positional[label] if mode == 'position_match' else label
        mask = parsed['library'][chosen]
        if mode == 'centers':
            offsets = [(0, 0)]
        else:
            mask = np.rot90(mask, 0 if mode == 'unturned_stencils' else k)
            offsets = [(int(rr) - t // 2, int(cc) - t // 2) for rr, cc in np.argwhere(mask)]
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < side and 0 <= cc < side):
                outside += 1
                continue
            if mode == 'issuing_piece' and assembly['owner'][rr, cc] != corner:
                continue
            counts[rr, cc] += 1
            possible_colors[list(map(int, parsed['key'])).index(label), rr, cc] = True
    return counts, possible_colors, outside


def _render(parsed, assembly, mode='correct'):
    if mode == 'solid':
        printed = np.ones_like(assembly['owner'], dtype=bool)
    else:
        counts, _, _ = _coverage(parsed, assembly, mode)
        printed = counts % 2 == 1 if mode == 'xor' else counts > 0
    return np.where(printed, parsed['key'][assembly['owner']], parsed['background']).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all commands, local frames, ownership, and colors from input only."""
    parsed = _parse(input_grid)
    return _render(parsed, _assembly(parsed))


def _witnesses(parsed, assembly):
    correct = _render(parsed, assembly)
    counts, colors, outside = _coverage(parsed, assembly)
    owner = assembly['owner']
    receiving_identity_present = np.take_along_axis(colors, owner[None], axis=0)[0]
    return dict(
        **{mode: not np.array_equal(correct, _render(parsed, assembly, mode))
           for mode in ('solid', 'centers', 'issuing_piece', 'unturned_stencils',
                        'position_match', 'frozen_commands', 'xor')},
        command_coloring=bool(np.any((counts > 0) & ~receiving_identity_present)),
        outer_clipping=outside > 0,
        blank_cells=int(np.sum(counts == 0)),
        overlap_cells=int(np.sum(counts >= 2)),
        cross_seam_cells=int(np.sum((counts > 0) & (_coverage(parsed, assembly, 'issuing_piece')[0] == 0))),
    )


def _key_locations(owner):
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)
            and any(min(max(abs(rr - r - dr), abs(cc - c - dc))
                        for dr in (0, 1) for dc in (0, 1)) >= 2
                    for rr, cc in np.argwhere(owner == 0))]


def _candidate_cut(rng, side, kind):
    if kind == 0:
        # Unequal horizontal bands with independently wandering vertical seams.
        y, x, z = map(int, rng.integers(2, side - 1, 3))
        owner = np.empty((side, side), dtype=np.int8)
        for r in range(side):
            cut = x if r < y else z
            owner[r, :cut], owner[r, cut:] = (0, 1) if r < y else (2, 3)
            if r < y and r >= 1:
                x = int(np.clip(x + rng.integers(-1, 2), 1, side - 1))
            elif r >= y:
                z = int(np.clip(z + rng.integers(-1, 2), 1, side - 1))
        for _ in range(side * 2):
            r, c = map(int, rng.integers(1, side - 1, 2))
            dr, dc = NEIGHBORS[int(rng.integers(4))]
            old = int(owner[r, c])
            owner[r, c] = owner[r + dr, c + dc]
            if any(len(_components(owner == i)) != 1 or np.sum(owner == i) < 5 for i in range(4)):
                owner[r, c] = old
        return owner
    if kind == 1:
        # Connected frontier growth with independently weighted piece areas.
        owner = np.full((side, side), -1, dtype=np.int8)
        owner[:2, :2], owner[:2, -2:] = 0, 1
        owner[-2:, :2], owner[-2:, -2:] = 2, 3
        weights = rng.uniform(.3, 2.7, 4)
        while np.any(owner == -1):
            frontier = []
            for r, c in np.argwhere(owner == -1):
                labels = {int(owner[r + dr, c + dc]) for dr, dc in NEIGHBORS
                          if 0 <= r + dr < side and 0 <= c + dc < side and owner[r + dr, c + dc] >= 0}
                frontier.extend((int(r), int(c), label) for label in sorted(labels))
            probs = np.array([weights[label] for _, _, label in frontier])
            r, c, label = frontier[int(rng.choice(len(frontier), p=probs / probs.sum()))]
            owner[r, c] = label
        return owner
    # A large wrapping piece with three smaller profiled corner pieces.
    large = int(rng.integers(4))
    owner = np.full((side, side), large, dtype=np.int8)
    for corner in rng.permutation([i for i in range(4) if i != large]):
        h, w = map(int, rng.integers(2, side - 1, 2))
        widths = rng.integers(1, w + 1, h)
        widths[0] = w
        if corner == 0:
            widths[:2] = np.maximum(widths[:2], 2)
        for d, width in enumerate(widths):
            r = d if corner < 2 else side - 1 - d
            for x in range(int(width)):
                c = x if corner % 2 == 0 else side - 1 - x
                if owner[r, c] != large:
                    return None
                owner[r, c] = corner
    return owner


def _partition(rng, side):
    for _ in range(240):
        owner = _candidate_cut(rng, side, int(rng.choice(3, p=[.55, .25, .20])))
        if owner is None or not _key_locations(owner):
            continue
        pieces = []
        for i in range(4):
            mask = owner == i
            if mask.sum() < 5 or len(_components(mask)) != 1:
                break
            small, _ = _crop(mask)
            rotations = {(np.rot90(small, k).shape, np.rot90(small, k).tobytes()) for k in range(4)}
            if len(rotations) != 4:
                break
            pieces.append(dict(mask=small, commands=()))
        if len(pieces) != 4:
            continue
        packs = _packings(pieces, side)
        if len(packs) == 1 and np.array_equal(packs[0]['owner'], owner):
            return owner
    raise RuntimeError("bounded cut search exhausted")


def _motif(rng, t, asymmetric=False):
    for _ in range(40):
        style = int(rng.integers(8))
        rr, cc = np.indices((t, t))
        center = t // 2
        if style == 0:
            mask = rng.random((t, t)) < rng.uniform(.18, .62)
        elif style == 1:
            mask = rr == center
        elif style == 2:
            mask = (rr == cc) | (rr + cc == t - 1)
        elif style == 3:
            mask = (rr == center) | (cc == center)
        elif style == 4:
            mask = (rr == 0) | (cc == 0) | (rr == t - 1) | (cc == t - 1)
        elif style == 5:
            mask = (rr + cc) % 2 == int(rng.integers(2))
        elif style == 6:
            mask = (rr <= center) & (cc >= center)
        else:
            mask = np.ones((t, t), dtype=bool)
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        mask[center, center] = True
        if not asymmetric or all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            return mask
    mask = np.zeros((t, t), dtype=bool)
    mask[center, center] = mask[center - 1, center] = mask[center - 1, center + 1] = True
    return mask


def _commands(rng, owner, key_location, count, identities):
    kr, kc = key_location
    patch = {(kr + dr, kc + dc) for dr in (0, 1) for dc in (0, 1)}
    selected = []
    # Each piece starts with one identity; four identities are always exercised.
    labels = list(map(int, rng.permutation(identities)))
    for corner in range(4):
        candidates = [tuple(map(int, p)) for p in np.argwhere(owner == corner)
                      if (corner != 0 or min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in patch) >= 2)]
        if not candidates:
            return None
        # Bias toward seams, but let commands occupy interiors and outer edges too.
        if rng.random() < .6:
            boundary = [p for p in candidates if any(
                0 <= p[0] + dr < len(owner) and 0 <= p[1] + dc < len(owner)
                and owner[p[0] + dr, p[1] + dc] != corner for dr, dc in NEIGHBORS)]
            candidates = boundary or candidates
        r, c = candidates[int(rng.integers(len(candidates)))]
        selected.append((r, c, labels[corner], corner))
    for raw in rng.permutation(np.argwhere(np.ones_like(owner, dtype=bool))):
        if len(selected) >= count:
            break
        r, c = map(int, raw)
        corner = int(owner[r, c])
        if corner == 0 and min(max(abs(r - q[0]), abs(c - q[1])) for q in patch) < 2:
            continue
        if any(corner == pc and max(abs(r - pr), abs(c - pq)) < 2 for pr, pq, _, pc in selected):
            continue
        selected.append((r, c, int(rng.choice(identities)), corner))
    return selected if len(selected) == count else None


def _scatter(rng, images, t, background):
    max_height, min_width = 30 - (t + 3), 4 * t + 5
    # Try unconstrained placements using footprint halos; boxes may interleave.
    if rng.random() < .45:
        for _ in range(8):
            h = int(rng.integers(max(max(a.shape[0] for a in images) + 2, 12), max_height + 1))
            w = int(rng.integers(min_width, 31))
            canvas = np.full((h, w), background, dtype=np.int8)
            forbidden = np.zeros((h, w), dtype=bool)
            success = True
            for i in rng.permutation(4):
                image = images[int(i)]
                coords = np.argwhere(image != background)
                ih, iw = image.shape
                for _ in range(60):
                    r = int(rng.integers(1, h - ih))
                    c = int(rng.integers(1, w - iw))
                    points = coords + (r, c)
                    if forbidden[points[:, 0], points[:, 1]].any():
                        continue
                    canvas[points[:, 0], points[:, 1]] = image[coords[:, 0], coords[:, 1]]
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            forbidden[points[:, 0] + dr, points[:, 1] + dc] = True
                    break
                else:
                    success = False
                    break
            if success:
                return canvas
    # All ordered shelf groupings, along either axis, independently of corner role.
    choices = []
    gap = int(rng.integers(1, 4))
    for transpose in (False, True):
        dims = [a.shape[::-1] if transpose else a.shape for a in images]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, i in enumerate(order):
                    group.append(i)
                    if j == 3 or (cuts >> j) & 1:
                        groups.append(group)
                        group = []
                widths = [sum(dims[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
                heights = [max(dims[i][0] for i in g) for g in groups]
                h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
                hh, ww = (w, h) if transpose else (h, w)
                if hh + 2 <= max_height and ww + 2 <= 30:
                    choices.append((transpose, dims, groups, heights, widths, hh, ww))
    if not choices:
        raise RuntimeError("no scatter layout fits")
    transpose, dims, groups, heights, widths, hh, ww = choices[int(rng.integers(len(choices)))]
    h = int(rng.integers(hh + 2, min(max_height, hh + 6) + 1))
    w = int(rng.integers(max(min_width, ww + 2), 31))
    canvas = np.full((h, w), background, dtype=np.int8)
    # Work in shelf axes and then exchange coordinates, not piece orientations.
    ah, aw = (w, h) if transpose else (h, w)
    extent_h, extent_w = (ww, hh) if transpose else (hh, ww)
    r0 = int(rng.integers(1, ah - extent_h))
    for g, gh, gw in zip(groups, heights, widths):
        c0 = int(rng.integers(1, aw - gw))
        for i in g:
            rr = r0 + int(rng.integers(gh - dims[i][0] + 1))
            row, col = (c0, rr) if transpose else (rr, c0)
            a = images[i]
            points = np.argwhere(a != background)
            canvas[points[:, 0] + row, points[:, 1] + col] = a[points[:, 0], points[:, 1]]
            c0 += dims[i][1] + gap
        r0 += gh + gap
    return canvas


def _generate_pair(rng, side, t, count, first=False, second=False):
    owner = _partition(rng, side)
    palette = list(map(int, rng.permutation(10)))
    bg, frame, identities = palette[0], palette[1], palette[2:6]
    carrier_pool = palette[6:6 + int(rng.integers(1, 5))]
    carriers = list(map(int, rng.choice(carrier_pool, 4)))
    library = {label: _motif(rng, t, asymmetric=first) for label in identities}
    if not first and rng.random() < .20:
        a, b = rng.choice(identities, 2, replace=False)
        library[int(b)] = library[int(a)].copy()
    order = tuple(map(int, rng.permutation(identities)))
    key = np.array(rng.permutation(identities), dtype=np.int8)
    if first and tuple(map(int, key)) == order:
        key = np.roll(key, 1)
    key_location = _key_locations(owner)[int(rng.integers(len(_key_locations(owner))))]
    commands = _commands(rng, owner, key_location, count, identities)
    if commands is None:
        raise RuntimeError("command placement exhausted")
    if first:
        input_turns = [0] + list(map(int, rng.permutation([1, 2, 3])))
    elif second:
        q = int(rng.integers(1, 4))
        input_turns = [0] + list(map(int, rng.permutation([0, q, q])))
    else:
        input_turns = [0] + list(map(int, rng.integers(0, 4, 3)))
    images = []
    for corner in range(4):
        mask, (r0, c0) = _crop(owner == corner)
        image = np.where(mask, carriers[corner], bg).astype(np.int8)
        for r, c, label, pc in commands:
            if pc == corner:
                image[r - r0, c - c0] = label
        if corner == 0:
            kr, kc = key_location
            image[kr - r0:kr - r0 + 2, kc - c0:kc - c0 + 2] = key.reshape(2, 2)
        images.append(np.rot90(image, input_turns[corner]).copy())
    scene = _scatter(rng, images, t, bg)
    grid = np.full((t + 3 + scene.shape[0], scene.shape[1]), bg, dtype=np.int8)
    width = 4 * t + 5
    grid[0, :width] = grid[t + 1, :width] = frame
    grid[:t + 2, :width:t + 1] = frame
    grid[t + 2] = frame
    for j, label in enumerate(order):
        col = 1 + j * (t + 1)
        grid[1:t + 1, col:col + t] = np.where(library[label], label, bg)
    grid[t + 3:] = scene
    # Construct the answer directly from the accepted owner grid and pre-scatter
    # command positions, rather than solving the just-built input.
    printed = np.zeros((side, side), dtype=bool)
    for r, c, label, corner in commands:
        stencil = np.rot90(library[label], -input_turns[corner])
        rr, cc = np.nonzero(stencil)
        rr, cc = rr + r - t // 2, cc + c - t // 2
        inside = (rr >= 0) & (rr < side) & (cc >= 0) & (cc < side)
        printed[rr[inside], cc[inside]] = True
    output = np.full((side, side), bg, dtype=np.int8)
    for corner in range(4):
        output[printed & (owner == corner)] = key[corner]
    return dict(input=grid, output=output)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Seeded generation with visible conventions and a bounded geometric search."""
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    preferred_t = int(rng.choice([3, 5]))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            if regime == 0:
                side = int(rng.integers(7, 9) if split == 'train' else rng.integers(9, 11))
            elif regime == 1:
                side = int(rng.integers(9, 11) if split == 'train' else rng.integers(7, 9))
            else:
                side = int(rng.integers(7, 11))
            t = 3 if split == 'train' and i == 0 else 5 if split == 'train' and i == 1 else (
                preferred_t if rng.random() < .7 else 8 - preferred_t)
            count = int(rng.integers(8, 12) if regime == 2 and split == 'test' else rng.integers(4, 8))
            for _ in range(100):
                try:
                    pair = _generate_pair(rng, side, t, count,
                                          first=split == 'train' and i == 0,
                                          second=split == 'train' and i == 1)
                except RuntimeError:
                    continue
                parsed = _parse(pair['input'])
                assembly = _assembly(parsed)
                _require(np.array_equal(pair['output'], _render(parsed, assembly)), "construction and input reconstruction disagree")
                witnesses = _witnesses(parsed, assembly)
                if witnesses['blank_cells'] < max(3, side * side // 10):
                    continue
                if split == 'train' and i == 0 and not all(witnesses[name] for name in (
                        'solid', 'centers', 'issuing_piece', 'unturned_stencils', 'position_match',
                        'frozen_commands', 'xor', 'command_coloring', 'outer_clipping')):
                    continue
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded example construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training_sizes, test_sizes, evidence = set(), set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            p = _parse(inp)
            a = _assembly(p)
            _require(out.shape == (p['side'], p['side']), "output area differs from footprint area")
            _require(np.array_equal(out, _render(p, a)), "incorrect printed surface")
            _require(set(map(int, np.unique(out))) == {p['background'], *map(int, p['key'])},
                     "all receiver colors and background must be visible")
            # Each center is printed in its receiving piece's key color; command
            # identity does not replace that corner-derived color.
            for r, c, _, _, corner in a['commands']:
                _require(a['owner'][r, c] == corner and out[r, c] == p['key'][corner], "command transport failure")
            w = _witnesses(p, a)
            _require(w['blank_cells'] >= max(3, p['side'] ** 2 // 10), "printing must leave visible blank cells")
            (training_sizes if split == 'train' else test_sizes).add(p['t'])
            if split == 'train':
                evidence.append(w)
    _require(test_sizes <= training_sizes, "test stencil scale requires demonstration")
    for name in ('solid', 'centers', 'issuing_piece', 'unturned_stencils', 'position_match',
                 'frozen_commands', 'xor', 'command_coloring', 'outer_clipping'):
        _require(any(w[name] for w in evidence), "missing training witness: " + name)


def measure_puzzle(puzzle):
    """Descriptive measurements from actual grids, not sampler labels."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p = _parse(pair['input'])
            a = _assembly(p)
            areas = [int(np.sum(a['owner'] == i)) for i in range(4)]
            fills, edge_counts = [], []
            for i in range(4):
                mask = a['owner'] == i
                small, _ = _crop(mask)
                fills.append(float(small.mean()))
                edge_counts.append(sum(bool(x.any()) for x in (mask[0], mask[-1], mask[:, 0], mask[:, -1])))
            origins = [piece['source_origin'] for piece in p['pieces']]
            reading = sorted(range(4), key=lambda i: origins[i])
            corner_by_piece = {index: corner for corner, (index, _) in enumerate(a['motions'])}
            records.append(dict(split=split, side=p['side'], stencil_side=p['t'],
                canvas=list(np.asarray(pair['input']).shape), areas=areas, area_ratio=max(areas) / min(areas),
                min_fill=min(fills), max_edge_contacts=max(edge_counts),
                commands=len(a['commands']), rotations=[k for _, k in a['motions']],
                background=p['background'], frame=p['frame'], carrier_count=len({x['carrier'] for x in p['pieces']}),
                library_order=list(p['order']), key=list(map(int, p['key'])),
                reading_corners=[corner_by_piece[i] for i in reading], key_location=list(p['pieces'][0]['key_origin']),
                motif_pixels=[int(x.sum()) for x in p['library'].values()],
                duplicate_stencils=len({x.tobytes() for x in p['library'].values()}) < 4,
                witnesses=_witnesses(p, a)))
    return records


def self_test():
    """Bounded 42-seed checks, count boundaries, and live corruption checks."""
    puzzles, records, shapes, cuts, motifs, reading_orders = [], [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(np.array(pair['input'], copy=True)), pair['output']), "solve mismatch")
                p = _parse(pair['input'])
                a = _assembly(p)
                shapes.add(np.asarray(pair['input']).shape)
                cuts.add((p['side'], a['owner'].tobytes()))
                motifs.update((m.shape, m.tobytes()) for m in p['library'].values())
        local = measure_puzzle(puzzle)
        records.extend(local)
        reading_orders.update(tuple(r['reading_corners']) for r in local)
    signature = lambda task: tuple((s, np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                                   np.asarray(p['output']).tobytes()) for s in ('train', 'test') for p in task[s])
    _require(len({signature(p) for p in puzzles}) == 42, "42 seeds must differ")
    _require(signature(puzzles[0]) == signature(generate_puzzle(0)), "seed reproducibility")
    _require({r['side'] for r in records} == {7, 8, 9, 10}, "assembly scale coverage")
    _require({r['stencil_side'] for r in records} == {3, 5}, "stencil scale coverage")
    _require(len(shapes) >= 20 and len(cuts) >= 120 and len(motifs) >= 60, "structural variation collapsed")
    _require(len(reading_orders) >= 18, "scattering must not encode corners")
    _require({r['background'] for r in records} == set(range(10)), "background palette coverage")
    _require(len({r['carrier_count'] for r in records}) >= 3, "carrier sharing coverage")
    _require(max(r['area_ratio'] for r in records) >= 4, "unequal piece sizes missing")
    _require(any(r['max_edge_contacts'] >= 3 for r in records), "wrapping cuts missing")
    _require(any(r['duplicate_stencils'] for r in records), "repeated stencil silhouettes missing")
    _require(max(r['commands'] for r in records) >= 9, "command count variation missing")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        q = generate_puzzle(9000 + a + b, a, b)
        _require(len(q['train']) == a and len(q['test']) == b, "count boundary failure")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Change only the visible legend. Geometry, commands, and printing stay fixed;
    # the two receiver colors must exchange everywhere in the output.
    original = puzzles[0]['train'][0]
    changed = np.array(original['input'], copy=True)
    p = _parse(changed)
    anchor = p['pieces'][0]
    r0, c0 = anchor['source_origin']
    kr, kc = anchor['key_origin']
    kr, kc = kr + r0 + p['t'] + 3, kc + c0
    changed_key = p['key'].copy()
    changed_key[0], changed_key[3] = changed_key[3], changed_key[0]
    changed[kr:kr + 2, kc:kc + 2] = changed_key.reshape(2, 2)
    expected = np.array(original['output'], copy=True)
    expected[np.asarray(original['output']) == p['key'][0]] = p['key'][3]
    expected[np.asarray(original['output']) == p['key'][3]] = p['key'][0]
    _require(np.array_equal(solve(changed), expected), "visible legend recoloring failure")
    # Carrier colors carry no corner information. Changing one to an unused
    # compatible carrier cannot affect the print.
    carrier_changed = np.array(original['input'], copy=True)
    used = set(map(int, np.unique(carrier_changed)))
    replacement = next(color for color in range(10) if color not in used)
    carrier_changed[carrier_changed == anchor['carrier']] = replacement
    _require(np.array_equal(solve(carrier_changed), original['output']), "carrier-color shortcut")
    # A command erased to background breaks square-area conservation.
    damaged_input = np.array(puzzles[0]['test'][0]['input'], copy=True)
    p = _parse(damaged_input)
    piece = p['pieces'][1]
    r, c, _ = piece['commands'][0]
    r0, c0 = piece['source_origin']
    damaged_input[p['t'] + 3 + r0 + r, c0 + c] = p['background']
    try:
        solve(damaged_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing footprint cell accepted")
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes), cuts=len(cuts),
                motifs=len(motifs), scatter_corner_orders=len(reading_orders),
                semantic_verification="generator-owned checks only")


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
