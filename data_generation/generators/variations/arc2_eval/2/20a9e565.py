"""Object-sequence continuation inside two inclusive, canonical L markers.

Labels are painted from latent objects. Validation segments the returned input,
fits arithmetic geometry and modular colors, and uses separate cell predicates.
No generated metadata is needed to solve a pair.
"""
import numpy as np


_FAMILIES = ('filled', 'rails', 'comb', 'hollow', 'zigzag',
             'F', 'ladder', 'chevron', 'wave', 'branch')


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + ' out of range')
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count('num_train', num_train, 2, 10)
    num_test = _checked_count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _place_markers(grid, bounds, marker):
    for r, c in _marker_cells(bounds):
        grid[r, c] = marker


def _find_marker(grid):
    found = []
    for raw in np.unique(grid):
        pos = np.argwhere(grid == raw)
        if len(pos) != 6:
            continue
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        if min(r1 - r0, c1 - c0) < 1 or r1 - r0 + c1 - c0 < 4:
            continue
        bounds = (r0, r1, c0, c1)
        if set(map(tuple, pos)) == _marker_cells(bounds):
            found.append((int(raw), bounds))
    assert len(found) == 1, 'ambiguous or malformed corner annotations'
    return found[0]


def _paint_object(family, h, w, aux, phase, palette, background):
    """Ground truth: explicit strokes/subunits, independent of fitting masks."""
    a = np.full((h, w), background, dtype=np.int8)
    color = palette[phase % len(palette)]
    if family == 'filled':
        a[:] = color
    elif family == 'rails':
        a[0] = color
        a[-1] = color
    elif family == 'hollow':
        a[0] = color
        a[-1] = color
        a[:, 0] = color
        a[:, -1] = color
    elif family in ('comb', 'ladder', 'F'):
        spacing = aux[0]
        a[:, 0] = color
        if family == 'ladder':
            a[:, -1] = color
        if family == 'F':
            for k in range(aux[1]):
                a[h - 1 - k * spacing] = color
        else:
            for r in range(0, h, spacing):
                a[r] = color
    elif family == 'zigzag':
        for r in range(h):
            k = r % (2 * w - 2)
            a[r, min(k, 2 * w - 2 - k)] = color
    elif family == 'chevron':
        spacing = aux[0]
        center = (h - 1) // 2
        units = (h - 3) // (2 * spacing) + 1
        for k in range(units):
            col = palette[(phase + k) % len(palette)]
            hand = w - 1 if k % 2 == 0 else 0
            for rr in {center - spacing * k, center + spacing * k}:
                a[rr] = col
                a[rr - 1, hand] = col
                a[rr + 1, hand] = col
    elif family == 'wave':
        wave_phase = aux[0]
        period = 2 * w - 1
        for r in range(h):
            k = (r + wave_phase * (w - 1)) % period
            c = min(k, period - 1 - k)
            a[r, c] = color
            if c + 1 < w:
                a[r, c + 1] = color
    elif family == 'branch':
        depth, spread, cap = aux
        units = w // depth
        center = (h - 1) // 2
        left, right = center - cap // 2, center + cap // 2
        for k in range(units):
            c = k * depth
            col = palette[(phase + k) % len(palette)]
            prev = palette[(phase + k - 1) % len(palette)]
            if k == 0:
                a[left:right + 1, c] = col
                a[left, c + 1:c + depth] = col
                a[right, c + 1:c + depth] = col
            else:
                lo, hi = left - spread * k, right + spread * k
                a[lo:lo + spread + 1, c] = col
                a[hi - spread:hi + 1, c] = col
                a[lo, c + 1:c + depth] = col
                a[hi, c + 1:c + depth] = col
                a[lo + spread, c + 1:c + depth] = prev
                a[hi - spread, c + 1:c + depth] = prev
    else:
        raise AssertionError('unknown object family')
    return a


def _cell_template(family, h, w, aux):
    """Input-side model: cell predicates return depth, or -1 for background."""
    if h < 1 or w < 1:
        return None
    r, c = np.indices((h, w))
    depth = np.zeros((h, w), dtype=int)
    if family == 'filled':
        mask = np.ones((h, w), dtype=bool)
    elif family == 'rails':
        mask = (r == 0) | (r == h - 1)
    elif family == 'hollow':
        mask = (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    elif family == 'comb':
        mask = (c == 0) | (r % aux[0] == 0)
    elif family == 'ladder':
        mask = (c == 0) | (c == w - 1) | (r % aux[0] == 0)
    elif family == 'F':
        spacing, bars = aux
        if h <= spacing * (bars - 1):
            return None
        mask = (c == 0) | (((h - 1 - r) % spacing == 0) &
                           (h - 1 - r <= spacing * (bars - 1)))
    elif family == 'zigzag':
        if w < 2 or h < w:
            return None
        phase = r % (2 * w - 2)
        mask = c == np.minimum(phase, 2 * w - 2 - phase)
    elif family == 'chevron':
        spacing = aux[0]
        if h < 3 or (h - 3) % (2 * spacing) or w < 2:
            return None
        distance = np.abs(r - (h - 1) // 2)
        # With spacing two, adjacent tips of opposite hands share a row.
        center_row = distance % spacing == 0
        inner = distance // spacing
        outer = inner + 1
        maxk = (h - 3) // (2 * spacing)
        hand_inner = np.where(inner % 2 == 0, w - 1, 0)
        hand_outer = np.where(outer % 2 == 0, w - 1, 0)
        mask_center = center_row & (inner <= maxk)
        mask_inner = (distance % spacing == 1) & (c == hand_inner) & (inner <= maxk)
        mask_outer = (distance % spacing == spacing - 1) & (c == hand_outer) & (outer <= maxk)
        mask = mask_center | mask_inner | mask_outer
        depth = np.where(mask_outer, outer, inner)
    elif family == 'wave':
        phase = aux[0]
        if w < 2 or h < 2 * w - 1 or (h - phase) % (2 * w - 1):
            return None
        p = 2 * w - 1
        t = (r + phase * (w - 1)) % p
        low = np.minimum(t, p - 1 - t)
        mask = (c == low) | (c == low + 1)
    elif family == 'branch':
        tier, spread, cap = aux
        if w % tier or w < tier or h != cap + 2 * spread * (w // tier - 1):
            return None
        k = c // tier
        within = c % tier
        mid = (h - 1) // 2
        radial = np.abs(r - mid)
        edge = cap // 2 + spread * k
        central = (k == 0) & (((within == 0) & (radial <= cap // 2)) |
                               (radial == cap // 2))
        later = (k > 0) & (((within == 0) & (radial >= edge - spread) &
                             (radial <= edge)) | (radial == edge) |
                            (radial == edge - spread))
        mask = central | later
        depth = k - ((k > 0) & (within > 0) & (radial == edge - spread))
    else:
        return None
    return np.where(mask, depth, -1)


def _linear(values):
    values = list(map(int, values))
    if len(values) < 3:
        return None
    delta = values[1] - values[0]
    if any(v != values[0] + i * delta for i, v in enumerate(values)):
        return None
    return values[0], delta


def _descriptors(patches):
    """Candidate explanations, not a bank of rendered layouts."""
    for family in ('branch', 'chevron', 'wave', 'F', 'ladder', 'comb',
                   'hollow', 'rails', 'zigzag', 'filled'):
        if family == 'branch':
            for tier in (2, 3):
                for spread in (1, 2, 3):
                    cap = patches[0].shape[0] - 2 * spread * (patches[0].shape[1] // tier - 1)
                    if cap in (3, 5, 7):
                        yield family, (tier, spread, cap)
        elif family in ('ladder', 'comb', 'chevron'):
            for spacing in (2, 3, 4) if family != 'chevron' else (2, 3):
                yield family, (spacing,)
        elif family == 'F':
            for spacing in (2, 3, 4):
                for bars in (2, 3, 4):
                    yield family, (spacing, bars)
        elif family == 'wave':
            for start in (0, 1):
                for step in (0, 1):
                    yield family, (start, step)
        else:
            yield family, ()


def _fit_colors(patches, templates, background):
    colors = sorted({int(v) for a in patches for v in np.unique(a) if v != background})
    p = len(colors)
    if not 1 <= p <= 4:
        return []
    fits = []
    compound = any(np.any(t > 0) for t in templates)
    if not compound and len(patches) < max(3, p + 1):
        return []
    for advance in (0, 1, -1):
        mapping = {}
        good = True
        for i, (a, t) in enumerate(zip(patches, templates)):
            for d in np.unique(t[t >= 0]):
                vals = np.unique(a[t == d])
                if len(vals) != 1:
                    good = False
                    break
                residue = (int(d) + advance * i) % p
                val = int(vals[0])
                if residue in mapping and mapping[residue] != val:
                    good = False
                    break
                mapping[residue] = val
            if not good:
                break
        if good and len(mapping) == p and set(mapping.values()) == set(colors):
            fits.append((advance, tuple(mapping[k] for k in range(p))))
    return fits


def _canonical_fits(grid, bounds, background):
    r0, r1, c0, c1 = bounds
    positions = np.argwhere(grid != background)
    if len(positions) == 0:
        return
    if not (positions[:, 1].max() < c0 or positions[:, 1].min() > c1):
        return
    cols = np.flatnonzero(np.any(grid != background, axis=0))
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    if not 3 <= len(groups) <= 9:
        return
    starts, tops, patches = [], [], []
    for group in groups:
        rows = np.flatnonzero(np.any(grid[:, group] != background, axis=1))
        starts.append(int(group[0]))
        tops.append(int(rows[0]))
        patches.append(grid[rows[0]:rows[-1] + 1, group[0]:group[-1] + 1])
    widths = [a.shape[1] for a in patches]
    heights = [a.shape[0] for a in patches]
    wl = _linear(widths)
    if wl is None:
        return
    gaps = [starts[i + 1] - starts[i] - widths[i] for i in range(len(groups) - 1)]
    if len(set(gaps)) == 1:
        gl = (gaps[0], 0)
    else:
        gl = _linear(gaps)
        if gl is None:
            return
    for flip_r in (False, True):
        for flip_c in (False, True):
            oriented = [a[::-1 if flip_r else 1, ::-1 if flip_c else 1] for a in patches]
            for family, aux in _descriptors(oriented):
                if family == 'wave':
                    phases = [(aux[0] + aux[1] * i) % 2 for i in range(len(patches))]
                    if len(patches) < 4 and aux[1]:
                        continue
                    units = [(h - ph) // (2 * widths[0] - 1) for h, ph in zip(heights, phases)]
                    hl = _linear(units)
                    if wl[1] or hl is None:
                        continue
                    object_aux = [(ph,) for ph in phases]
                else:
                    hl = _linear(heights)
                    if hl is None:
                        continue
                    object_aux = [aux] * len(patches)
                templates = []
                for a, arg in zip(oriented, object_aux):
                    t = _cell_template(family, *a.shape, arg)
                    if t is None or not np.array_equal(t >= 0, a != background):
                        break
                    templates.append(t)
                if len(templates) != len(patches):
                    continue
                colors = _fit_colors(oriented, templates, background)
                if not colors:
                    continue
                for align in (0, 1, 2):
                    anchor = _linear([2 * y + align * (h - 1) for y, h in zip(tops, heights)])
                    if anchor is None:
                        continue
                    for advance, palette in colors:
                        model = dict(family=family, aux=aux, width=wl, height=hl,
                                     gap=gl, start=starts[0], anchor=anchor, align=align,
                                     flip_r=flip_r, flip_c=flip_c, advance=advance,
                                     palette=palette, visible=len(patches), tops=tops,
                                     widths=widths, heights=heights, starts=starts)
                        predicted = _predict_model(model, bounds, background)
                        if predicted is not None:
                            yield predicted, model


def _geometry(model, i):
    w = model['width'][0] + i * model['width'][1]
    h = model['height'][0] + i * model['height'][1]
    aux = model['aux']
    if model['family'] == 'wave':
        phase = (aux[0] + i * aux[1]) % 2
        h = h * (2 * w - 1) + phase
        aux = (phase,)
    x = (model['start'] + i * (model['width'][0] + model['gap'][0]) +
         i * (i - 1) // 2 * (model['width'][1] + model['gap'][1]))
    doubled_top = model['anchor'][0] + i * model['anchor'][1] - model['align'] * (h - 1)
    if doubled_top % 2:
        return None
    return doubled_top // 2, x, h, w, aux


def _predict_model(model, bounds, background):
    r0, r1, c0, c1 = bounds
    prediction = np.full((r1 - r0 + 1, c1 - c0 + 1), background, dtype=np.int8)
    # Start with the observed interval and extend toward the requested window.
    # Stop before any non-positive hypothetical object; never jump over one.
    indices = list(range(model['visible']))
    if c0 > model['starts'][-1]:
        i = model['visible']
        while i < 40:
            geom = _geometry(model, i)
            if geom is None:
                return None
            y, x, h, w, aux = geom
            if h < 1 or w < 1 or model['gap'][0] + (i - 1) * model['gap'][1] < 1:
                return None
            if x > c1:
                break
            indices.append(i)
            i += 1
    else:
        i = -1
        while i > -40:
            geom = _geometry(model, i)
            if geom is None:
                return None
            y, x, h, w, aux = geom
            if h < 1 or w < 1 or model['gap'][0] + i * model['gap'][1] < 1:
                return None
            if x + w - 1 < c0:
                break
            indices.append(i)
            i -= 1
    for i in indices:
        geom = _geometry(model, i)
        if geom is None:
            return None
        y, x, h, w, aux = geom
        if x > c1 or x + w <= c0 or y > r1 or y + h <= r0:
            continue
        if h > 150 or w > 150:
            return None
        t = _cell_template(model['family'], h, w, aux)
        if t is None:
            return None
        t = t[::-1 if model['flip_r'] else 1, ::-1 if model['flip_c'] else 1]
        lo_r, hi_r = max(r0, y), min(r1 + 1, y + h)
        lo_c, hi_c = max(c0, x), min(c1 + 1, x + w)
        region = t[lo_r - y:hi_r - y, lo_c - x:hi_c - x]
        target = prediction[lo_r - r0:hi_r - r0, lo_c - c0:hi_c - c0]
        mask = region >= 0
        colors = np.asarray(model['palette'], dtype=np.int8)
        target[mask] = colors[(region[mask] + i * model['advance']) % len(colors)]
    if not np.any(prediction != background):
        return None
    return prediction


def _analyze(grid):
    marker, bounds = _find_marker(grid)
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])
    assert background != marker and np.count_nonzero(counts == max(counts)) == 1
    r0, r1, c0, c1 = bounds
    clean = grid.copy()
    clean[clean == marker] = background
    assert np.all(clean[r0:r1 + 1, c0:c1 + 1] == background), 'window must contain only annotations'
    answers = []
    for transpose in (False, True):
        a = clean.T if transpose else clean
        bb = (c0, c1, r0, r1) if transpose else bounds
        for prediction, model in _canonical_fits(a, bb, background):
            pred = prediction.T.copy() if transpose else prediction
            answers.append((pred, model, transpose, bb))
    assert answers, 'visible objects do not establish a supported continuation'
    expected = answers[0][0]
    assert all(np.array_equal(expected, a[0]) for a in answers), 'competing visible models predict different crops'
    _, model, transpose, bb = answers[0]
    forward = bb[2] > model['starts'][-1]
    # Actual requested objects and their phases, obtained from fitted input only.
    target_indices = []
    domain = range(model['visible'], 40) if forward else range(-1, -40, -1)
    for i in domain:
        geom = _geometry(model, i)
        if geom is None:
            break
        y, x, h, w, _ = geom
        gap_index = i - 1 if forward else i
        if min(h, w, model['gap'][0] + gap_index * model['gap'][1]) < 1:
            break
        if (forward and x > bb[3]) or (not forward and x + w <= bb[2]):
            break
        if x <= bb[3] and x + w > bb[2] and y <= bb[1] and y + h > bb[0]:
            target_indices.append(i)
    assert target_indices
    nearest = min(target_indices) if forward else max(target_indices)
    y, x, h, w, _ = _geometry(model, nearest)
    skips = nearest - model['visible'] if forward else -nearest - 1
    target_geometry = [_geometry(model, i) for i in target_indices]
    canonical = clean.T if transpose else clean
    prefix = canonical[bb[0]:bb[1] + 1, model['start']:model['start'] + bb[3] - bb[2] + 1]
    canonical_answer = expected.T if transpose else expected
    info = dict(background=background, marker=marker, bounds=bounds,
                family=model['family'], direction=('down' if forward else 'up') if transpose else
                ('right' if forward else 'left'), axis='vertical' if transpose else 'horizontal',
                visible=model['visible'], palette=list(model['palette']),
                cycle_length=len(model['palette']), compound=model['family'] in ('chevron', 'branch') and
                len(model['palette']) > 1, height_delta=model['height'][1], width_delta=model['width'][1],
                visible_heights=list(model['heights']), visible_widths=list(model['widths']),
                top_offsets=list(model['tops']), axial_starts=list(model['starts']),
                motif_parameters=list(model['aux']),
                growing=bool(model['height'][1] or model['width'][1]),
                transverse_step=model['anchor'][1] / 2, gap=model['gap'][0], gap_delta=model['gap'][1],
                skipped=skips, crop_phase=(nearest * model['advance']) % len(model['palette']),
                crop_gap=int(bb[2] - model['starts'][-1] - model['widths'][-1]) if forward else
                int(model['starts'][0] - bb[3] - 1),
                axial_cut=any(bb[2] > xx or bb[3] < xx + ww - 1 for yy, xx, hh, ww, aa in target_geometry),
                transverse_cut=any(bb[0] > yy or bb[1] < yy + hh - 1 for yy, xx, hh, ww, aa in target_geometry),
                prefix_copy=bool(np.array_equal(prefix, canonical_answer)),
                target_objects=len(target_indices), output_shape=list(expected.shape),
                output_colors=len(set(map(int, np.unique(expected))) - {background}),
                bottom_blank=bool(np.all(expected[-1] == background)),
                right_blank=bool(np.all(expected[:, -1] == background)),
                candidate_models=len(answers))
    return expected, info


def solve(train_pairs, test_input):
    """Predict from copies of demonstrations and the test input, never labels."""
    for pair in train_pairs:
        prediction, _ = _analyze(pair['input'])
        assert np.array_equal(prediction, pair['output']), 'inconsistent marker/continuation convention'
    return _analyze(test_input)[0]
_SOURCE_CASES = (((30, 30),
  ((22, '000000000000010000000000000000'),
   (23, '000000004000010000000000000000'),
   (24, '000020004000010000000000000000'),
   (25, '010020004000010000000000000000'),
   (26, '011022204444011111055000000000'),
   (27, '010020004000010000050000500000'),
   (28, '011022204444011111000005500000')),
  ('222222', '200000', '222222')),
 ((29, 29),
  ((3, '00000000000000004000000000000'),
   (4, '00000000000000004400000000000'),
   (5, '00000000000000804800000000000'),
   (6, '00000000000008808800000000000'),
   (7, '00000000009009809800000000000'),
   (8, '00000000009909909900000000000'),
   (9, '00000000409409409400000000000'),
   (10, '00000004404404404400000000000'),
   (11, '00008008408408408400000000000'),
   (12, '00008808808808808800000000000'),
   (13, '00908908908908908900000000000'),
   (14, '09909909909909909905500000000'),
   (15, '00908908908908908905000000000'),
   (16, '00008808808808808800000000000'),
   (17, '00008008408408408400000000000'),
   (18, '00000004404404404400000000000'),
   (19, '00000000409409409400000000000'),
   (20, '00000000009909909900000000000'),
   (21, '00000000009009809800000000000'),
   (22, '00000000000008808800000000000'),
   (23, '00000000000000804800000000000'),
   (24, '00000000000000004400000000000'),
   (25, '00000000000000004000000000000'),
   (26, '00000000000000000000500000000'),
   (27, '00000000000000000005500000000')),
  ('99', '89', '88', '84', '44', '94', '99', '98', '88', '48', '44', '49', '99', '09')),
 ((30, 30),
  ((1, '077702220333077702220000055000'),
   (2, '070702020303070702020000050000'),
   (3, '077702220333077702220000000000'),
   (4, '070702020303070702020000000000'),
   (5, '077702220333077702220000000000'),
   (6, '000002020303070702020000000000'),
   (7, '000002220333077702220000000000'),
   (8, '000000000303070702020000000000'),
   (9, '000000000333077702220000000000'),
   (10, '000000000000070702020000000000'),
   (11, '000000000000077702220000000000'),
   (12, '000000000000000002020000000000'),
   (13, '000000000000000002220000000000'),
   (16, '000000000000000000000000000500'),
   (17, '000000000000000000000000005500')),
  ('777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777',
   '707',
   '777')),
 ((30, 30),
  ((11, '000000055000000000000000000500'),
   (12, '000000050000000000000000005500'),
   (14, '000004000440004400044000400000'),
   (15, '000004404444044440444404400000'),
   (16, '000000444004440044400444000000'),
   (18, '000000000660006600066000000000'),
   (19, '000000006666066660666600000000'),
   (20, '000000066006660066600660000000'),
   (22, '000000000040004400040000000000'),
   (23, '000000000044044440440000000000'),
   (24, '000000000004440044400000000000'),
   (26, '000000000000006600000000000000'),
   (27, '000000000000066660000000000000'),
   (28, '000000000000660066000000000000')),
  ('066660666606666066660', '660066600666006660066')),
 ((29, 29),
  ((0, '00000000000033333000000000000'),
   (1, '00000000000030003000000000000'),
   (3, '00000000000022222000000000000'),
   (4, '00000000000020002000000000000'),
   (5, '00000000003330003330000000000'),
   (6, '00000000003020002030000000000'),
   (8, '00000000000033333000000000000'),
   (9, '00000000000030003000000000000'),
   (10, '00000000002220002220000000000'),
   (11, '00000000002030003020000000000'),
   (12, '00000000333000000033300000000'),
   (13, '00000000302000000020300000000'),
   (15, '00000055000000000000000000000'),
   (16, '00000050000000000000000000000'),
   (21, '00000000000000000000005000000'),
   (22, '00000000000000000000055000000')),
  ('00000022222000000',
   '00000020002000000',
   '00003330003330000',
   '00003020002030000',
   '00222000000022200',
   '00203000000030200',
   '33300000000000333',
   '30200000000000203')))


def _sample_conventions(rng):
    level = int(rng.choice([0, 1, 2], p=[0.18, 0.40, 0.42]))
    roles = list(map(int, rng.permutation(10)))
    background = 0 if rng.random() < 0.65 else roles[0]
    roles.remove(background)
    marker = roles.pop()
    rng.shuffle(roles)
    challenge = str(rng.choice(['axis', 'distance', 'scale', 'palette', 'combined']))
    isolated = level > 0 and challenge != 'combined' and rng.random() < 0.45
    regime = int(rng.integers(4))
    if isolated:
        regime = int(rng.integers(2)) if challenge == 'axis' else 2
    return dict(level=level, background=background, marker=marker, palette=tuple(roles),
                varying_roles=level > 0 and rng.random() < 0.6,
                regime=regime, axis=int(rng.integers(2)),
                challenge=challenge, isolated=isolated, shared_flip=bool(rng.integers(2)),
                simple_family=str(rng.choice(_FAMILIES[:5] + ('ladder',))))


def _sample_pair(rng, params, regime, index):
    level = params['level']
    test = regime == 'target_test'
    isolated = params['isolated']
    if isolated and params['challenge'] == 'scale':
        family = str(rng.choice(_FAMILIES[:5]))
    elif level == 0:
        family = params['simple_family'] if rng.random() < 0.65 else str(rng.choice(_FAMILIES[:5]))
    else:
        family = str(rng.choice(_FAMILIES))
    background, marker = params['background'], params['marker']
    palette_pool = params['palette']
    if params['varying_roles'] and not isolated:
        values = list(map(int, rng.permutation(10)))
        background = 0 if rng.random() < 0.4 else values[0]
        values.remove(background)
        marker = values.pop()
        palette_pool = tuple(values)
    p = 2 if level == 0 or isolated else int(rng.integers(1, 5))
    if test and params['challenge'] in ('palette', 'combined') and level:
        p = int(rng.integers(3, 5))
    if family == 'branch':
        p = min(p, 3)
    palette = palette_pool[:p]
    phase = int(rng.integers(p))
    palette = palette[phase:] + palette[:phase]
    compound = family in ('chevron', 'branch')
    advance = int(rng.integers(2)) if compound else 1
    nvisible = int(rng.integers(3, 6)) if compound else int(rng.integers(max(3, p + 1), 7))
    if family == 'branch':
        nvisible = 3
    skip = 0 if level == 0 else int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
    if test and level and params['challenge'] in ('distance', 'combined'):
        skip = int(rng.integers(1, 3))
    if isolated and params['challenge'] == 'distance' and not test:
        skip = 0
    span = int(rng.integers(1, 4)) if family not in ('F', 'branch') else int(rng.integers(1, 3))
    total = nvisible + skip + span
    grow = level > 0 and (rng.random() < 0.73 or (not test and index == 0))
    if isolated and params['challenge'] == 'scale':
        grow = False
    delta = int(rng.choice([-1, 1, 1, 2])) if grow else 0
    spacing = int(rng.integers(2, 5))
    aux = ()
    h0, dh = int(rng.integers(3, 10)), delta
    w0, dw = int(rng.integers(1 if family == 'filled' else 2, 5)), 0
    if family in ('filled', 'hollow', 'rails', 'comb') and grow and rng.random() < 0.3:
        dw = 1
    if family in ('comb', 'ladder'):
        aux = (spacing,)
    if family == 'ladder':
        w0 = int(rng.integers(3, 5))
        h0 = 1 + spacing * int(rng.integers(2, 5))
        dh = spacing * delta
    elif family == 'zigzag':
        h0 = int(rng.integers(2 * w0, 12))
    elif family == 'F':
        bars = int(rng.integers(2, 5))
        aux = (spacing, bars)
        h0 = spacing * (bars - 1) + int(rng.integers(2, 6))
        w0 = int(rng.integers(2, 4))
        dw = 1 if grow and rng.random() < 0.8 else 0
    elif family == 'chevron':
        spacing = int(rng.integers(2, 4))
        aux = (spacing,)
        w0 = int(rng.integers(2, 4))
        h0 = 3 + 2 * spacing * int(rng.integers(0, 2))
        dh = 2 * spacing * (1 if delta > 0 else delta)
    elif family == 'branch':
        depth = int(rng.integers(2, 4))
        spread = int(rng.integers(1, 4))
        cap = int(rng.choice([3, 5, 7]))
        aux = (depth, spread, cap)
        units0 = 1 if delta >= 0 else total
        du = 1 if delta > 0 else delta
        h0, dh = cap + 2 * spread * (units0 - 1), 2 * spread * du
        w0, dw = depth * units0, depth * du
    elif family == 'wave':
        w0 = int(rng.integers(2, 5))
        h0 = int(rng.integers(1, 3))  # unit count, not bounding height
        dh = 1 if delta > 0 else delta
        aux = (int(rng.integers(2)), int(rng.integers(2)))
        nvisible = max(nvisible, 4)
        total = nvisible + skip + span
    if isolated and params['challenge'] == 'scale':
        h0 = int(rng.integers(9, 15)) if test else int(rng.integers(4, 8))
    elif test and params['challenge'] in ('scale', 'combined') and family in _FAMILIES[:5] + ('F',):
        h0 += int(rng.integers(1, 4))
    if family != 'branch' and dh < 0:
        h0 += (total - 1) * -dh
    if family == 'wave':
        heights = [(h0 + i * dh) * (2 * w0 - 1) + (aux[0] + i * aux[1]) % 2 for i in range(total)]
    else:
        heights = [h0 + i * dh for i in range(total)]
    widths = [w0 + i * dw for i in range(total)]
    if min(heights + widths) < 1 or max(heights + widths) > 65:
        return None
    align = int(rng.integers(3))
    if len({h % 2 for h in heights}) > 1 and align == 1:
        align = int(rng.choice([0, 2]))
    drift = 0 if level == 0 else int(rng.choice([-2, -1, 0, 0, 0, 1, 2]))
    gap = int(rng.integers(1, 4))
    gap_delta = 1 if level > 0 and nvisible >= 4 and w0 <= 2 and rng.random() < 0.15 else 0
    objects = []
    x = 0
    for i, (h, w) in enumerate(zip(heights, widths)):
        y = i * drift - align * (h - 1) // 2
        arg = ((aux[0] + i * aux[1]) % 2,) if family == 'wave' else aux
        a = _paint_object(family, h, w, arg, advance * i, palette, background)
        if not np.any(a != background) or not np.all(np.any(a != background, axis=0)):
            return None
        if not np.any(a[0] != background) or not np.any(a[-1] != background):
            return None
        objects.append((y, x, a))
        x += w + gap + i * gap_delta
    targets = objects[nvisible + skip:]
    first_y, first_x, first = targets[0]
    last_y, last_x, last = targets[-1]
    crosslo = min(y for y, x, a in targets)
    crosshi = max(y + len(a) - 1 for y, x, a in targets)
    # Four crop modes vary independently of the construction and split.
    mode = int(rng.integers(4))
    if mode == 0:
        r0, r1 = crosslo, crosshi
    elif mode == 1:
        r0 = int(rng.integers(crosslo, max(crosslo + 1, crosshi)))
        r1 = min(crosshi, r0 + int(rng.integers(2, 10)) - 1)
    elif mode == 2:
        r1 = crosshi
        r0 = max(crosslo, r1 - int(rng.integers(2, 8)) + 1)
    else:
        r0, r1 = crosslo - int(rng.integers(3)), crosshi + int(rng.integers(3))
    c0 = first_x + (int(rng.integers(first.shape[1])) if rng.random() < 0.45 else 0)
    c1 = last_x + last.shape[1] - 1
    if rng.random() < 0.3:
        c1 -= int(rng.integers(last.shape[1]))
    elif rng.random() < 0.2:
        c1 += int(rng.integers(gap + 1))
    if r1 - r0 < 1:
        r0 = r1 - 1
    if c1 - c0 < 1:
        c1 = c0 + 1
    if r1 - r0 + c1 - c0 < 4:
        r0 -= 4 - (r1 - r0 + c1 - c0)
    visible = objects[:nvisible]
    ymin = min([r0] + [y for y, x, a in visible])
    ymax = max([r1] + [y + len(a) - 1 for y, x, a in visible])
    xmin = min(0, c0)
    xmax = max([c1] + [x + a.shape[1] - 1 for y, x, a in visible])
    height, width = ymax - ymin + 1, xmax - xmin + 1
    if max(height, width) > 30:
        return None
    # Padding is optional, and 30x30 grids remain legal.
    top = int(rng.integers(min(3, 30 - height) + 1))
    bottom = int(rng.integers(min(4, 30 - height - top) + 1))
    left = int(rng.integers(min(3, 30 - width) + 1))
    right = int(rng.integers(min(4, 30 - width - left) + 1))
    canvas = np.full((height + top + bottom, width + left + right), background, dtype=np.int8)
    dr, dc = top - ymin, left - xmin
    for y, x, a in visible:
        canvas[y + dr:y + dr + len(a), x + dc:x + dc + a.shape[1]] = a
    bb = (r0 + dr, r1 + dr, c0 + dc, c1 + dc)
    answer = np.full((r1 - r0 + 1, c1 - c0 + 1), background, dtype=np.int8)
    for y, x, a in objects:
        for rr, cc in np.argwhere(a != background):
            gr, gc = y + int(rr), x + int(cc)
            if r0 <= gr <= r1 and c0 <= gc <= c1:
                assert answer[gr - r0, gc - c0] == background, 'overlapping objects'
                answer[gr - r0, gc - c0] = a[rr, cc]
    if not np.any(answer != background):
        return None
    # Reflect content and the crop, then redraw canonical annotations.
    flip_r = bool(rng.integers(2))
    flip_c = params['shared_flip'] if isolated else (bool(rng.integers(2)) if level else False)
    if flip_r:
        canvas = canvas[::-1].copy()
        answer = answer[::-1].copy()
        bb = (len(canvas) - 1 - bb[1], len(canvas) - 1 - bb[0], bb[2], bb[3])
    if flip_c:
        canvas = canvas[:, ::-1].copy()
        answer = answer[:, ::-1].copy()
        bb = (bb[0], bb[1], canvas.shape[1] - 1 - bb[3], canvas.shape[1] - 1 - bb[2])
    shift = params['regime']
    if level == 0 or shift == 2:
        axis = params['axis']
    elif shift in (0, 1):
        axis = int(test) if shift == 0 else int(not test)
    else:
        axis = int(rng.integers(2))
    if axis:
        canvas = canvas.T.copy()
        answer = answer.T.copy()
        bb = (bb[2], bb[3], bb[0], bb[1])
    _place_markers(canvas, bb, marker)
    try:
        prediction, info = _analyze(canvas)
    except AssertionError:
        return None
    if not np.array_equal(prediction, answer):
        return None
    if grow and not info['growing']:
        return None
    return {'input': canvas, 'output': answer}


def _grid_key(grid):
    return grid.shape, tuple(map(int, grid.flat))


def _source_pairs():
    pairs = []
    for shape, rows, output in _SOURCE_CASES:
        a = np.zeros(shape, dtype=np.int8)
        for r, line in rows:
            a[r] = [int(v) for v in line]
        b = np.asarray([[int(v) for v in line] for line in output], dtype=np.int8)
        pairs.append({'input': a, 'output': b})
    return pairs


_SOURCE_KEYS = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}


def _fallback_pair(params, variant, test):
    """A finite pool of 32 distinct, directly constructible legal pairs."""
    height, top = 3 + variant % 8, 1 + variant // 8
    bg, marker = params['background'], params['marker']
    a, b = params['palette'][:2]
    grid = np.full((height + top + 1, 12), bg, dtype=np.int8)
    for x, color in ((1, a), (3, b), (5, a)):
        grid[top:top + height, x] = color
    out = np.full((height, 4), bg, dtype=np.int8)
    out[:, 0], out[:, 2] = b, a
    bounds = (top, top + height - 1, 7, 10)
    axis = params['axis']
    if params['level'] and params['regime'] in (0, 1):
        axis = int(test) if params['regime'] == 0 else int(not test)
    if axis:
        grid, out = grid.T.copy(), out.T.copy()
        bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
    _place_markers(grid, bounds, marker)
    return {'input': grid, 'output': out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    params = _sample_conventions(rng)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count, regime in (('train', num_train, 'source_train'), ('test', num_test, 'target_test')):
        for index in range(count):
            pair = None
            for attempt in range(152):
                candidate = (_sample_pair(rng, params, regime, index) if attempt < 120 else
                             _fallback_pair(params, attempt - 120, split == 'test'))
                if candidate is None:
                    continue
                key = _grid_key(candidate['input'])
                if key in seen or (key, _grid_key(candidate['output'])) in _SOURCE_KEYS:
                    continue
                pair = candidate
                break
            assert pair is not None, 'bounded construction exhausted'
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, pairs = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= v <= 30 for v in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            ik, ok = _grid_key(pair['input']), _grid_key(pair['output'])
            assert ik not in seen and (ik, ok) not in pairs and (ik, ok) not in _SOURCE_KEYS
            assert ik != ok
            seen.add(ik)
            pairs.add((ik, ok))
            predicted, info = _analyze(pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
            assert info['marker'] not in pair['output']
            assert set(np.unique(pair['output'])) <= set(np.unique(pair['input']))
            assert info['visible'] >= 3 and info['skipped'] >= 0
    # Every pair exposes its own arithmetic and color parameters. The shared
    # convention (inclusive L crop with markers replaced) is checked above in
    # every demonstration; no fixed style, palette, axis, or split is assumed.
    train = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
    for pair in puzzle['test']:
        assert np.array_equal(solve(train, pair['input'].copy()), pair['output'])


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test() -> None:
    # These regressions use the original grid CONTENTS, including the skipped
    # ladder and the backward, alternating-phase wave. They are never generated.
    source = _source_pairs()
    for pair in source:
        assert np.array_equal(_analyze(pair['input'])[0], pair['output'])
    assert _analyze(source[2]['input'])[1]['skipped'] == 1
    assert _analyze(source[3]['input'])[1]['direction'] == 'up'
    counts_train, counts_test, keys, shapes = set(), set(), set(), set()
    families, directions, backgrounds, markers, palettes = set(), set(), set(), set(), set()
    family_counts = {family: 0 for family in _FAMILIES}
    visible_counts, cycles, spacings, parameters, aspect_classes, axis_regimes = set(), set(), set(), set(), set(), set()
    distinct_width_growth = set()
    totals = dict(growth=0, shrink=0, compound=0, skipped=0, partial=0,
                  multicolor=0, multibody=0, phase=0, displacement=0, gap_growth=0,
                  bottom_content=0, right_content=0, combined=0,
                  within_family=0, within_palette=0, within_roles=0, within_axis=0,
                  within_growth=0, within_scale=0, within_offset=0, within_crop=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        keys.add(_puzzle_key(puzzle))
        infos = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, info = _analyze(pair['input'])
                infos.append(info)
                families.add(info['family'])
                family_counts[info['family']] += 1
                visible_counts.add(info['visible'])
                cycles.add(info['cycle_length'])
                spacings.add(info['gap'])
                parameters.add((info['family'], tuple(info['motif_parameters'])))
                distinct_width_growth.add(info['width_delta'])
                oh, ow = pair['output'].shape
                aspect_classes.add('wide' if ow > 2 * oh else 'tall' if oh > 2 * ow else 'square')
                directions.add(info['direction'])
                backgrounds.add(info['background'])
                markers.add(info['marker'])
                palettes.add(tuple(info['palette']))
                shapes.add((pair['input'].shape, pair['output'].shape))
                totals['growth'] += info['growing']
                totals['shrink'] += info['height_delta'] < 0 or info['width_delta'] < 0
                totals['compound'] += info['compound']
                totals['skipped'] += info['skipped'] > 0
                totals['partial'] += info['axial_cut'] or info['transverse_cut']
                totals['multicolor'] += info['output_colors'] > 1
                totals['multibody'] += info['target_objects'] > 1
                totals['phase'] += info['crop_phase'] > 0
                totals['displacement'] += info['transverse_step'] != 0
                totals['gap_growth'] += info['gap_delta'] != 0
                totals['bottom_content'] += not info['bottom_blank']
                totals['right_content'] += not info['right_blank']
                totals['combined'] += info['growing'] and info['compound'] and (info['skipped'] > 0 or info['transverse_cut'])
        for label, prop in [('family', 'family'), ('palette', 'palette'), ('roles', 'background'), ('axis', 'axis')]:
            totals['within_' + label] += len({str(i[prop]) for i in infos}) > 1
        for label, prop in [('growth', 'growing'), ('scale', 'visible_heights'),
                            ('offset', 'top_offsets'), ('crop', 'output_shape')]:
            totals['within_' + label] += len({str(i[prop]) for i in infos}) > 1
        nt = len(puzzle['train'])
        axis_regimes.add((tuple(sorted({i['axis'] for i in infos[:nt]})),
                         tuple(sorted({i['axis'] for i in infos[nt:]}))))
    assert len(counts_train) >= 4 and len(counts_test) >= 3 and len(keys) >= 40
    assert len(shapes) >= 100 and len(families) == 10
    assert min(family_counts.values()) >= 3, family_counts
    assert len(directions) == 4 and len(backgrounds) >= 5 and len(markers) >= 5 and len(palettes) >= 30
    assert len(visible_counts) >= 3 and cycles == {1, 2, 3, 4}
    assert len(spacings) >= 3 and len(parameters) >= 20 and len(distinct_width_growth) >= 3
    assert aspect_classes == {'wide', 'tall', 'square'}
    assert {(('horizontal',), ('vertical',)), (('vertical',), ('horizontal',)),
            (('horizontal',), ('horizontal',)), (('vertical',), ('vertical',))} <= axis_regimes
    minimums = dict(growth=40, shrink=15, compound=10, skipped=20, partial=80,
                    multicolor=40, multibody=30, phase=40, displacement=40, gap_growth=1,
                    bottom_content=100, right_content=100, combined=5,
                    within_family=25, within_palette=20, within_roles=5, within_axis=10,
                    within_growth=15, within_scale=30, within_offset=30, within_crop=30)
    assert all(totals[k] >= value for k, value in minimums.items()), totals
    for seed, nt, ne in [(401, 2, 1), (402, 10, 1), (403, 10, 4),
                         (404, None, 1), (405, None, 4), (406, 2, None), (407, 10, None)]:
        first, second = generate_puzzle(seed, nt, ne), generate_puzzle(seed, nt, ne)
        assert _puzzle_key(first) == _puzzle_key(second)
        if nt is not None:
            assert len(first['train']) == nt
        if ne is not None:
            assert len(first['test']) == ne
        validate_generated_puzzle(first)
    assert _puzzle_key(generate_puzzle(88)) == _puzzle_key(generate_puzzle(88))
    # The fallback pool is larger than the maximum number of preceding pairs;
    # its distinctness makes exhaustion impossible even after unlucky proposals.
    convention = _sample_conventions(np.random.default_rng(18))
    fallback_keys = set()
    for variant in range(32):
        pair = _fallback_pair(convention, variant, False)
        fallback_keys.add(_grid_key(pair['input']))
        assert np.array_equal(_analyze(pair['input'])[0], pair['output'])
    assert len(fallback_keys) == 32
    for nt, ne in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3, 1.5)]:
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        else:
            raise AssertionError('illegal pair counts accepted')
    # A changed label must not evade the validator.
    p = generate_puzzle(902, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
