"""An invisible searchlight activates witnesses of a damaged mirror tapestry.

Only standard-library modules and NumPy are used.  Construction uses explicit
geometry and a clean tapestry; solve reads the returned input afresh.
"""
from __future__ import annotations

import numbers
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    """Eight-neighbor components, suitable for thin diagonal emitter rails."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, component = [start], [start]
        while queue:
            r, c = queue.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        queue.append(q)
                        component.append(q)
        result.append(component)
    return result


def _ring(shape, box):
    r0, r1, c0, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1 + 1, [c0, c1]] = True
    mask[[r0, r1], c0:c1 + 1] = True
    return mask


def _box(mask):
    p = np.argwhere(mask)
    return (int(p[:, 0].min()), int(p[:, 0].max()),
            int(p[:, 1].min()), int(p[:, 1].max()))


def _orbits(box):
    r0, r1, c0, c1 = box
    seen = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (r0 + r1 - r, c),
                                  (r, c0 + c1 - c),
                                  (r0 + r1 - r, c0 + c1 - c)}))
            seen.update(orbit)
            yield orbit


def _fit_stair(values, sign):
    """Fit the entire visible rail, including the cap endpoint."""
    values = np.array(values, dtype=int)
    if np.ptp(values) < 2:
        return None
    fits = []
    for period in (1, 2, 3):
        for phase in range(period):
            expected = values[0] + sign * ((np.arange(len(values)) + phase) // period)
            if np.array_equal(values, expected):
                fits.append((period, phase))
    return fits[0] if len(fits) == 1 else None


def _decode(value):
    """Find the contextual frame and the unique outward-pointing emitter."""
    inp = _grid(value)
    bg = int(inp[0, 0])
    if not (np.all(inp[[0, -1], :] == bg) and np.all(inp[:, [0, -1]] == bg)):
        raise ValueError("the outer border must identify one background/erasure color")
    matches = []
    for color in np.unique(inp):
        frame = int(color)
        if frame == bg:
            continue
        box = _box(inp == frame)
        r0, r1, c0, c1 = box
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        if not np.array_equal(inp == frame, _ring(inp.shape, box)):
            continue
        exterior = np.ones(inp.shape, dtype=bool)
        exterior[r0:r1 + 1, c0:c1 + 1] = False
        extra = set(map(int, np.unique(inp[exterior]))) - {bg, frame}
        if len(extra) != 1:
            continue
        solid = extra.pop()
        for component in _components(inp == solid):
            if len(component) < 7:
                continue
            component_mask = np.zeros(inp.shape, dtype=bool)
            for point in component:
                component_mask[point] = True
            for rotation in range(4):
                a = np.rot90(inp, -rotation)
                emitter = np.rot90(component_mask, -rotation)
                panel = _box(a == frame)
                _, _, rear, mouth = _box(emitter)
                if mouth - rear < 2 or mouth >= panel[2] - 1:
                    continue
                cap = np.flatnonzero(emitter[:, rear])
                if len(cap) < 3 or np.any(np.diff(cap) != 1):
                    continue
                upper, lower = [int(cap[0])], [int(cap[-1])]
                for x in range(rear + 1, mouth + 1):
                    rows = np.flatnonzero(emitter[:, x])
                    if len(rows) != 2 or rows[1] - rows[0] < 2:
                        break
                    upper.append(int(rows[0]))
                    lower.append(int(rows[1]))
                else:
                    top_fit = _fit_stair(upper, -1)
                    bottom_fit = _fit_stair(lower, 1)
                    if top_fit is None or bottom_fit is None:
                        continue
                    raw = np.zeros(a.shape, dtype=bool)
                    ys = np.arange(a.shape[0])
                    for x in range(mouth + 1, a.shape[1]):
                        hi = upper[0] - (x - rear + top_fit[1]) // top_fit[0]
                        lo = lower[0] + (x - rear + bottom_fit[1]) // bottom_fit[0]
                        raw[:, x] = (ys > hi) & (ys < lo)
                    # Scan the actual input. Transparent pattern/frame cells do
                    # not change blocked status; only the emitter's solid color does.
                    lit = np.zeros_like(raw)
                    for y in range(a.shape[0]):
                        blocked = False
                        for x in range(mouth + 1, a.shape[1]):
                            if not raw[y, x]:
                                continue
                            if a[y, x] == solid:
                                blocked = True
                            elif not blocked:
                                lit[y, x] = True
                    matches.append(dict(a=a, rotation=rotation, bg=bg, frame=frame,
                                        solid=solid, panel=panel, emitter=emitter,
                                        rear=rear, mouth=mouth, raw=raw, lit=lit,
                                        upper=tuple(upper), lower=tuple(lower),
                                        periods=(top_fit[0], bottom_fit[0]),
                                        phases=(top_fit[1], bottom_fit[1])))
    if len(matches) != 1:
        raise ValueError("input must contain one frame and one readable searchlight")
    return matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair an orbit iff an ORIGINAL surviving payload witness is illuminated."""
    d = _decode(input_grid)
    a, bg, solid = d['a'], d['bg'], d['solid']
    out = a.copy()
    for orbit in _orbits(d['panel']):
        witnesses = [p for p in orbit if int(a[p]) not in (bg, solid)]
        values = {int(a[p]) for p in witnesses}
        if len(values) > 1:
            raise ValueError("surviving mirror witnesses disagree")
        holes = [p for p in orbit if a[p] == bg]
        if holes and not witnesses:
            raise ValueError("every damaged orbit needs a surviving payload witness")
        if any(d['lit'][p] for p in witnesses):
            color = next(iter(values))
            for p in holes:
                out[p] = color
    return np.rot90(out, d['rotation']).copy()


def _geometry(rng, safe=False):
    """Construct one integrated scene in the right-facing coordinate system."""
    if safe:
        ph, pw = int(rng.integers(13, 17)), int(rng.integers(12, 17))
        periods, phases = (2, 2), (0, 1)
        length, rear, gap, py, center = 5, 1, 2, 3, 9
        height, width = ph + 6, pw + 12
        starts = (1, 2)
    else:
        ph, pw = (int(v) for v in rng.integers(8, 19, size=2))
        periods = tuple(int(v) for v in rng.integers(1, 4, size=2))
        if rng.random() < .45:
            periods = (periods[0], periods[0])
        phases = tuple(int(rng.integers(p)) for p in periods)
        length = 2 * max(periods) + int(rng.integers(2))
        rear = int(rng.integers(1, 4))
        gap = int(rng.integers(1, 4))
        starts = tuple(int(v) for v in rng.integers(1, 3, size=2))
        mouth = rear + length
        min_width = mouth + gap + pw + 4
        if min_width > 30:
            pw = 30 - mouth - gap - 4
            min_width = 30
        if pw < 8:
            return None
        width = int(rng.integers(min_width, 31))
        top_span = starts[0] + (length + phases[0]) // periods[0]
        bottom_span = starts[1] + (length + phases[1]) // periods[1]
        min_height = max(ph + 4, top_span + bottom_span + 3)
        height = int(rng.integers(min_height, 31))
        py = int(rng.integers(1, height - ph - 2))
        preferred = py + ph // 2 + int(rng.integers(-4, 5))
        center = int(np.clip(preferred, top_span + 1, height - bottom_span - 2))
    mouth = rear + length
    px = mouth + gap + 1
    panel = (py, py + ph + 1, px, px + pw + 1)
    emitter = np.zeros((height, width), dtype=bool)
    emitter[center - starts[0]:center + starts[1] + 1, rear] = True
    for i in range(1, length + 1):
        emitter[center - starts[0] - (i + phases[0]) // periods[0], rear + i] = True
        emitter[center + starts[1] + (i + phases[1]) // periods[1], rear + i] = True
    raw = np.zeros_like(emitter)
    ys = np.arange(height)
    for x in range(mouth + 1, width):
        top = center - starts[0] - (x - rear + phases[0]) // periods[0]
        bottom = center + starts[1] + (x - rear + phases[1]) // periods[1]
        raw[:, x] = (ys > top) & (ys < bottom)
    return dict(panel=panel, emitter=emitter, raw=raw, rear=rear, mouth=mouth,
                periods=periods, phases=phases)


def _field(rng, shape, box, colors, style):
    result = np.zeros(shape, dtype=np.int8)
    r0, r1, c0, c1 = box
    k = len(colors)
    bh, bw = (int(v) for v in rng.integers(1, 4, size=2))
    offset = int(rng.integers(k))
    coarse = rng.integers(k, size=(r1 - r0, c1 - c0))
    for orbit in _orbits(box):
        r, c = orbit[0]
        y, x = r - r0 - 1, c - c0 - 1
        if style == 0:
            index = int(rng.integers(k))
        elif style == 1:
            index = (y // bh + x // bw + offset) % k
        elif style == 2:
            index = (max(y // bh, x // bw) + offset) % k
        elif style == 3:
            index = int(coarse[y // bh, x // bw])
        else:
            index = ((2 * y + x) // bw + offset) % k
        if style and rng.random() < .08:
            index = int(rng.integers(k))
        for p in orbit:
            result[p] = colors[index]
    return result


def _obstacles(rng, g, mode, safe=False):
    """Detached solid groups; their cells may overlay the transparent tapestry."""
    emitter, raw, box = g['emitter'], g['raw'], g['panel']
    h, w = emitter.shape
    ring = _ring(emitter.shape, box)
    forbidden = ring.copy()
    forbidden[[0, -1], :] = True
    forbidden[:, [0, -1]] = True
    for y, x in np.argwhere(emitter):
        forbidden[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    blocks = np.zeros_like(emitter)
    if mode != 'clean':
        if safe:
            # A before-panel transverse bar gives symmetric partner rows
            # different visibility. This is ordinary family geometry.
            blocks[8:10, g['mouth'] + 1] = True
            if mode == 'test':
                blocks[8, 12] = True
        else:
            shapes = (((0, 0),), ((0, 0), (1, 0)),
                      ((0, 0), (1, 0), (2, 0)),
                      ((0, 0), (0, 1), (1, 0), (1, 1)),
                      ((0, 0), (1, 0), (1, 1)))
            groups = int(rng.integers(1, 3) if mode == 'blocked' else rng.integers(2, 5))
            for index in range(groups):
                offsets = shapes[int(rng.integers(1, len(shapes)))] if index == 0 else shapes[int(rng.integers(len(shapes)))]
                options = []
                for x in range(g['mouth'] + 1, min(box[3], box[2] + (box[3] - box[2]) // 2 + 1)):
                    for y in range(1, h - 1):
                        points = [(y + dy, x + dx) for dy, dx in offsets]
                        if any(yy >= h - 1 or xx >= w - 1 or forbidden[yy, xx]
                               or blocks[yy, xx] for yy, xx in points):
                            continue
                        if any(raw[p] for p in points):
                            options.append(points)
                if options:
                    for p in options[int(rng.integers(len(options)))]:
                        blocks[p] = True
            # A repeated obstacle tests union of straight lane shadows.
            if mode == 'test':
                options = [(int(y), x) for y, first in np.argwhere(blocks & raw)
                           for x in range(int(first) + 2, box[3])
                           if raw[y, x] and not forbidden[y, x] and not blocks[y, x]]
                if options:
                    blocks[options[int(rng.integers(len(options)))]] = True
    # Inert dots use the same solid color as effective blockers.
    for _ in range(int(rng.integers(3))):
        options = np.argwhere(~raw & ~forbidden & ~blocks)
        if len(options):
            p = tuple(options[int(rng.integers(len(options)))])
            blocks[p] = True
    shadow = np.zeros_like(raw)
    for y, x in np.argwhere(blocks & raw):
        shadow[y, x + 1:] = True
    lit = raw & ~shadow & ~blocks
    return blocks, lit


def _holes(rng, box, blocks, style):
    mask = np.zeros(blocks.shape, dtype=bool)
    r0, r1, c0, c1 = box
    h, w = r1 - r0 - 1, c1 - c0 - 1
    if style == 0:
        mask[r0 + 1:r1, c0 + 1:c1] = rng.random((h, w)) < rng.uniform(.12, .32)
    else:
        for _ in range(int(rng.integers(2, 6))):
            rh, cw = int(rng.integers(1, min(5, h) + 1)), int(rng.integers(1, min(5, w) + 1))
            y, x = int(rng.integers(r0 + 1, r1 - rh + 1)), int(rng.integers(c0 + 1, c1 - cw + 1))
            if style == 1:
                mask[y:y + rh, x:x + cw] = True
            elif style == 2:
                mask[y:y + rh, x] = True
                mask[y, x:x + cw] = True
            elif style == 3:
                for dy in range(rh):
                    mask[y + dy, x:x + 1 + dy * (cw - 1) // max(1, rh - 1)] = True
            else:
                mask[y:y + rh, x:x + cw] |= rng.random((rh, cw)) < .55
    mask[blocks] = False
    for orbit in _orbits(box):
        available = [p for p in orbit if not blocks[p]]
        if available and all(mask[p] for p in available):
            mask[available[int(rng.integers(len(available)))]] = False
    return mask


def _build_pair(rng, colors, rotation, mode, field_style, mask_style, safe=False):
    bg, frame, solid = map(int, colors[:3])
    g = _geometry(rng, safe)
    if g is None:
        return None
    box = g['panel']
    blocks, lit = _obstacles(rng, g, mode, safe)
    orbits = list(_orbits(box))
    mixed = [o for o in orbits if len(o) == 4 and not any(blocks[p] for p in o)
             and any(lit[p] for p in o) and any(not lit[p] for p in o)]
    inactive_options = mixed if mode == 'clean' else [o for o in mixed
                         if any(g['raw'][p] and not lit[p] for p in o)]
    if len(mixed) < 3 or not inactive_options:
        return None
    inactive = inactive_options[int(rng.integers(len(inactive_options)))]
    active_options = [o for o in mixed if o != inactive]
    active = active_options[int(rng.integers(len(active_options)))]
    other_options = [o for o in orbits if len(o) >= 2 and o not in (inactive, active)
                     and not any(blocks[p] for p in o) and any(lit[p] for p in o)]
    if not other_options:
        return None
    other = other_options[int(rng.integers(len(other_options)))]
    clean = _field(rng, blocks.shape, box, colors[3:], field_style)
    # Two informative orbits visibly carry different payload colors.
    for orbit, color in ((active, colors[3]), (inactive, colors[3]),
                         (other, colors[4])):
        for p in orbit:
            clean[p] = color
    mask = _holes(rng, box, blocks, mask_style)
    survivor_options = [p for p in inactive if not lit[p] and
                        (mode == 'clean' or g['raw'][p])]
    survivor = survivor_options[int(rng.integers(len(survivor_options)))]
    for p in inactive:
        mask[p] = p != survivor
    survivor_options = [p for p in active if lit[p]]
    survivor = survivor_options[int(rng.integers(len(survivor_options)))]
    for p in active:
        mask[p] = p != survivor
    survivor_options = [p for p in other if lit[p]]
    survivor = survivor_options[int(rng.integers(len(survivor_options)))]
    for p in other:
        mask[p] = False
    targets = [p for p in other if p != survivor]
    mask[targets[int(rng.integers(len(targets)))]] = True
    inp = np.full(blocks.shape, bg, dtype=np.int8)
    r0, r1, c0, c1 = box
    inp[r0 + 1:r1, c0 + 1:c1] = clean[r0 + 1:r1, c0 + 1:c1]
    inp[_ring(inp.shape, box)] = frame
    inp[g['emitter'] | blocks] = solid
    inp[mask] = bg
    # Construction labels use the clean pattern and explicit geometric exposure;
    # they do not call solve or recover geometry from a label difference.
    out = inp.copy()
    for orbit in orbits:
        awake = any(lit[p] and not mask[p] and not blocks[p] for p in orbit)
        if awake:
            for p in orbit:
                if mask[p]:
                    out[p] = clean[p]
    return {'input': np.rot90(inp, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _measure(inp, out):
    """Input-derived counterfactuals and actual structural descriptors."""
    d = _decode(inp)
    a = d['a']
    b = np.rot90(np.asarray(out), -d['rotation'])
    bg, solid, lit, raw = d['bg'], d['solid'], d['lit'], d['raw']
    changed = a != b
    panel_mask = np.zeros(a.shape, dtype=bool)
    r0, r1, c0, c1 = d['panel']
    panel_mask[r0 + 1:r1, c0 + 1:c1] = True
    gaps = panel_mask & (a == bg)
    all_repair, target_gate, no_shadow, color_global = (a.copy() for _ in range(4))
    lit_colors = set(map(int, a[panel_mask & lit & (a != bg) & (a != solid)]))
    diagonal_only = 0
    for orbit in _orbits(d['panel']):
        witnesses = [p for p in orbit if int(a[p]) not in (bg, solid)]
        values = {int(a[p]) for p in witnesses}
        assert len(values) <= 1, 'contradictory mirror witnesses'
        holes = [p for p in orbit if a[p] == bg]
        assert not holes or witnesses, 'unrecoverable damaged orbit'
        if not holes:
            continue
        color = next(iter(values))
        for p in holes:
            all_repair[p] = color
            if color in lit_colors:
                color_global[p] = color
            if lit[p]:
                target_gate[p] = color
            if any(raw[q] for q in witnesses):
                no_shadow[p] = color
            if changed[p]:
                r, c = p
                direct = ((r0 + r1 - r, c), (r, c0 + c1 - c))
                if all(int(a[q]) in (bg, solid) for q in direct):
                    diagonal_only += 1
            # Separately check each target by its three reflected input lookups.
            reflected = {(r0 + r1 - p[0], p[1]), (p[0], c0 + c1 - p[1]),
                         (r0 + r1 - p[0], c0 + c1 - p[1])} - {p}
            donors = [q for q in reflected if int(a[q]) not in (bg, solid) and lit[q]]
            expected = int(a[donors[0]]) if donors else bg
            assert int(b[p]) == expected, 'incorrect witness activation'
    assert np.all(~changed | gaps), 'only tapestry gaps may change'
    assert np.array_equal(a[~gaps], b[~gaps]), 'visible material changed'
    active_colors = set(map(int, b[changed]))
    effective_blocks = raw & (a == solid)
    er0, er1, ec0, ec1 = _box(d['emitter'])
    return dict(shape=tuple(inp.shape), direction=d['rotation'],
                roles=(bg, d['frame'], solid), panel=(r1 - r0 - 1, c1 - c0 - 1),
                periods=d['periods'], phases=d['phases'],
                outline=_signature(d['emitter'][er0:er1 + 1, ec0:ec1 + 1]),
                gaps=int(gaps.sum()), changed=int(changed.sum()),
                inactive=int((gaps & ~changed).sum()),
                active_dark=int((changed & ~lit).sum()),
                inactive_lit=int((gaps & ~changed & lit).sum()),
                active_shadow=int((changed & raw & ~lit).sum()),
                diagonal_only=diagonal_only, active_colors=len(active_colors),
                no_beam_difference=int((all_repair != b).sum()),
                target_gate_difference=int((target_gate != b).sum()),
                color_global_difference=int((color_global != b).sum()),
                no_shadow_difference=int((no_shadow != b).sum()),
                effective_blocks=int(effective_blocks.sum()),
                internal_blocks=int((effective_blocks & panel_mask).sum()),
                inert_blocks=int(((a == solid) & ~d['emitter'] & ~raw).sum()),
                repeated_lanes=int((effective_blocks.sum(axis=1) > 1).sum()),
                mask_signature=_signature(gaps[r0 + 1:r1, c0 + 1:c1]),
                palette=len(np.unique(a[panel_mask & ~gaps & (a != solid)])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            assert np.array_equal(solve(inp), out), 'input-only solve disagrees'
            f = _measure(inp, out)
            assert f['changed'] >= 4 and f['active_colors'] >= 2
            assert f['inactive'] > 0 and f['active_dark'] > 0 and f['inactive_lit'] > 0
            assert f['diagonal_only'] > 0, 'both mirrors must matter'
            assert f['no_beam_difference'] > 0 and f['target_gate_difference'] > 0
            assert f['color_global_difference'] > 0, 'color-wide activation shortcut fits'
            if split == 'test':
                assert f['repeated_lanes'] > 0 and f['no_shadow_difference'] > 0
                assert f['active_shadow'] > 0
            records[split].append(f)
    assert any(f['effective_blocks'] == 0 for f in records['train']), 'missing clean illumination demonstration'
    assert any(f['no_shadow_difference'] > 0 and f['active_shadow'] > 0
               for f in records['train']), 'missing donor occlusion / remote repair demonstration'
    assert len({f['panel'] for f in records['train']}) > 1, 'training geometry collapsed'


def _count(name, value, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f'{name} must be in {lower}..{upper}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ns:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer')
    # Negative integer seeds are also deterministic; nonnegative seeds go to
    # default_rng unchanged, including integers larger than 64 bits.
    seed_value = int(seed)
    rng = np.random.default_rng(seed_value if seed_value >= 0 else seed_value % (1 << 64))
    shared_roles = bool(rng.random() < .25)
    base_palette = rng.permutation(10)
    regime = int(rng.integers(3))  # Shared, mixed, or an input-readable held-out rotation.
    preferred = int(rng.integers(4))
    mask_order = rng.permutation(5)
    held_out_mask = int(mask_order[0])
    training_masks = [int(v) for v in mask_order[1:]]
    puzzle, seen = {'train': [], 'test': []}, set()
    training_panels = set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            mode = ('clean' if index == 0 else 'blocked') if split == 'train' else 'test'
            if split == 'train' and index >= 2:
                mode = str(rng.choice(['clean', 'blocked']))
            rotation = (preferred if regime == 0 else
                        ((preferred + (split == 'test')) % 4 if regime == 2 else int(rng.integers(4))))
            palette = base_palette.copy() if shared_roles else rng.permutation(10)
            k = int(rng.integers(2, 8))
            if shared_roles:
                rng.shuffle(palette[3:])
            colors = palette[:3 + k]
            field_style = int(rng.integers(5))
            mask_style = (held_out_mask if split == 'test' else
                          (training_masks[index] if index < 2 else int(rng.choice(training_masks))))
            for attempt in range(128):
                pair = _build_pair(rng, colors, rotation, mode, field_style, mask_style, safe=attempt >= 96)
                if pair is None:
                    continue
                key = _signature(pair['input'])
                if key in seen:
                    continue
                try:
                    f = _measure(pair['input'], pair['output'])
                except (AssertionError, ValueError):
                    continue
                if not (f['active_dark'] and f['inactive_lit'] and f['diagonal_only']
                        and f['changed'] >= 4 and f['active_colors'] >= 2):
                    continue
                if mode != 'clean' and not (f['no_shadow_difference'] and f['active_shadow']):
                    continue
                if mode == 'test' and not f['repeated_lanes']:
                    continue
                if split == 'train' and index == 1 and f['panel'] in training_panels:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train':
                    training_panels.add(f['panel'])
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded semantic, count, determinism, equivariance and 42-seed checks."""
    # Hand-built phase fixtures. At panel column 8 the upper conceptual
    # boundary is row 5 (phase 0) or row 4 (phase 1). A donor ON that boundary
    # cannot activate its orbit; the neighboring inner donor can. Labels are
    # written directly, without construction or optical-mask helpers.
    for upper, edge in (((8, 8, 7, 7, 6), 5), ((8, 7, 7, 6, 6), 4)):
        for lower in ((10, 10, 11, 11, 12), (10, 11, 11, 12, 12)):
            a = np.zeros((22, 22), dtype=np.int8)
            a[3:17, 8:18] = 3
            a[2:18, [7, 18]] = 2
            a[[2, 17], 7:19] = 2
            a[8:11, 1] = 1
            for x in range(2, 6):
                a[upper[x - 1], x] = a[lower[x - 1], x] = 1
            inactive = ((edge, 8), (edge, 17), (19 - edge, 8), (19 - edge, 17))
            active = ((edge + 1, 8), (edge + 1, 17), (18 - edge, 8), (18 - edge, 17))
            for p in inactive:
                a[p] = 4 if p == inactive[0] else 0
            for p in active[1:]:
                a[p] = 0
            expected = a.copy()
            for p in active[1:]:
                expected[p] = 3
            assert np.array_equal(solve(a), expected), 'thin-boundary phase regression'
    # Exercise the termination geometry explicitly; normal sampling need not
    # reach it during the diversity suite.
    for mode in ('clean', 'blocked', 'test'):
        guard_rng = np.random.default_rng(881)
        for _ in range(8):
            q = _build_pair(guard_rng, np.arange(10, dtype=np.int8), 1, mode, 2, 3, safe=True)
            if q is not None:
                assert np.array_equal(solve(q['input']), q['output'])
                f = _measure(q['input'], q['output'])
                if mode == 'clean' or (f['no_shadow_difference'] and f['active_shadow']):
                    break
        else:
            raise AssertionError('termination geometry failed its rule checks')
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_signature(q[role]) for s in ('train', 'test')
                             for q in p[s] for role in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in p[split]:
                records.append((split, _measure(pair['input'], pair['output'])))
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p
                       for a, b in zip(p[s], again[s]) for k in ('input', 'output'))
    fs = [f for _, f in records]
    assert len(signatures) == 42
    assert len({f['shape'] for f in fs}) >= 50
    assert len({f['panel'] for f in fs}) >= 40
    assert len({f['outline'] for f in fs}) >= 40
    assert len({f['mask_signature'] for f in fs}) >= 150
    assert len({f['roles'] for f in fs}) >= 80
    assert {p for f in fs for p in f['periods']} == {1, 2, 3}
    assert {(h % 2, w % 2) for h, w in (f['panel'] for f in fs)} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert any(f['periods'][0] != f['periods'][1] for f in fs)
    assert sum(f['internal_blocks'] > 0 for f in fs) >= 10
    assert sum(f['repeated_lanes'] > 0 for f in fs) >= 5
    assert sum(f['inert_blocks'] > 0 for f in fs) >= 20
    assert len({f['gaps'] for f in fs}) >= 25
    for split in ('train', 'test'):
        assert {f['direction'] for s, f in records if s == split} == {0, 1, 2, 3}
    sample = generate_puzzle(0, 2, 1)
    for pair in sample['train'] + sample['test']:
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair['input'], turns)), np.rot90(pair['output'], turns))
        for flip in (np.fliplr, np.flipud):
            assert np.array_equal(solve(flip(pair['input'])), flip(pair['output']))
        # A color permutation must move semantic roles, not change the rule.
        perm = np.array([6, 2, 9, 0, 7, 3, 1, 8, 4, 5], dtype=np.int8)
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    for nt, ns in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Labels missing either an active repair or preserving an inactive hole
    # must both fail. The latter specifically tests the new exposure condition.
    for corrupt_active in (True, False):
        damaged = {s: [{k: v.copy() for k, v in q.items()} for q in sample[s]] for s in sample}
        pair = damaged['test'][0]
        if corrupt_active:
            point = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][point] = pair['input'][point]
        else:
            d = _decode(pair['input'])
            out = np.rot90(pair['output'], -d['rotation']).copy()
            for orbit in _orbits(d['panel']):
                gaps = [p for p in orbit if d['a'][p] == d['bg']]
                witnesses = [p for p in orbit if int(d['a'][p]) not in (d['bg'], d['solid'])]
                if gaps and not any(d['lit'][p] for p in witnesses):
                    out[gaps[0]] = d['a'][witnesses[0]]
                    break
            pair['output'] = np.rot90(out, d['rotation']).copy()
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            continue
        raise AssertionError('corrupt label accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
