"""Register exceptional crop edits on an atlas; coincident toggles cancel.

All semantic information is visible in each input. No parent data is imported.
"""
from collections import Counter

import numpy as np


def _transform(a, index):
    operations = (lambda x: x, lambda x: np.rot90(x),
                  lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3),
                  np.fliplr, np.flipud, lambda x: x.T,
                  lambda x: x.T[::-1, ::-1])
    return operations[index](a).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    component.append(point)
                    pending.append(point)
        result.append(component)
    return result


def _parse_with_frame(a, background, frame):
    occupied = np.zeros(a.shape, dtype=bool)
    canvases, packets = [], []
    for component in _components(a == frame):
        rr, cc = zip(*component)
        r, c, h, w = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
        if h < 4 or w < 4 or r == 0 or c == 0 or r+h == a.shape[0] or c+w == a.shape[1]:
            raise ValueError("frames need an exterior margin")
        box = a[r:r+h, c:c+w]
        rows = np.flatnonzero(np.all(box == frame, axis=1))
        cols = np.flatnonzero(np.all(box == frame, axis=0))
        if (len(rows) < 2 or len(cols) < 2 or rows[0] != 0 or cols[0] != 0
                or rows[-1] != h-1 or cols[-1] != w-1):
            raise ValueError("incomplete rectangular frame")
        expected = np.zeros(box.shape, dtype=bool)
        expected[rows, :] = True
        expected[:, cols] = True
        if not np.array_equal(expected, box == frame) or np.any(box == background):
            raise ValueError("frame interiors or separators are malformed")
        if np.any(occupied[r:r+h, c:c+w]):
            raise ValueError("frames overlap")
        occupied[r:r+h, c:c+w] = True
        ny, nx = len(rows)-1, len(cols)-1
        if nx == ny == 1:
            canvases.append(box[1:-1, 1:-1].copy())
            continue
        if not ((ny == 1 and 3 <= nx <= 6) or (nx == 1 and 3 <= ny <= 6)):
            raise ValueError("a report is a strip of three to six panels")
        if np.any(np.diff(rows) != rows[1]) or np.any(np.diff(cols) != cols[1]):
            raise ValueError("report panels must have equal dimensions")
        panels = [box[y+1:z, x+1:t].copy()
                  for y, z in zip(rows, rows[1:]) for x, t in zip(cols, cols[1:])]
        if min(panels[0].shape) < 2:
            raise ValueError("crop sides must be at least two")
        packets.append(panels)
    if len(canvases) != 1 or not packets or np.any(a[~occupied] != background):
        raise ValueError("need one atlas, reports, and an empty exterior")
    atlas = canvases[0]
    colors = tuple(map(int, np.unique(atlas)))
    if len(colors) != 2 or any(np.any(~np.isin(p, colors)) for row in packets for p in row):
        raise ValueError("atlas and reports must share two interior colors")
    return dict(atlas=atlas, packets=packets, colors=colors,
                background=background, frame=frame)


def _parse_scene(a):
    background = int(a[0, 0])
    if (np.any(a[0] != background) or np.any(a[-1] != background)
            or np.any(a[:, 0] != background) or np.any(a[:, -1] != background)):
        raise ValueError("the exterior border must be background")
    candidates = []
    for frame in sorted(set(map(int, np.unique(a))) - {background}):
        try:
            candidates.append(_parse_with_frame(a, background, frame))
        except ValueError:
            pass
    if len(candidates) != 1:
        raise ValueError("the frame role must be uniquely visible")
    return candidates[0]


def _registrations(atlas, patch):
    """Exact matches, including every orientation, role exchange and offset."""
    result = []
    for index in range(8):
        oriented = _transform(patch, index)
        h, w = oriented.shape
        if h > atlas.shape[0] or w > atlas.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(atlas, (h, w))
        for swap in (0, 1):
            candidate = oriented if swap == 0 else 1-oriented
            rows, cols = np.nonzero(np.all(windows == candidate, axis=(-1, -2)))
            result.extend((index, swap, int(r), int(c)) for r, c in zip(rows, cols))
    return result


def _analyze_input(input_grid):
    info = _parse_scene(_grid(input_grid))
    atlas = info['atlas']
    binary = (atlas == info['colors'][1]).astype(np.int8)
    multiplicity = np.zeros(atlas.shape, dtype=int)
    records = []
    for panels in info['packets']:
        counts = Counter(_key(p) for p in panels)
        master_key, count = counts.most_common(1)[0]
        if count != len(panels)-1 or len(counts) != 2:
            raise ValueError("each report needs a repeated crop and one visible exception")
        master = next(p for p in panels if _key(p) == master_key)
        column = next(i for i, p in enumerate(panels) if _key(p) != master_key)
        variant = panels[column]
        patch = (master == info['colors'][1]).astype(np.int8)
        if len(np.unique(patch)) != 2:
            raise ValueError("a locating crop must show both roles")
        matches = _registrations(binary, patch)
        if len(matches) != 1:
            raise ValueError("the repeated crop needs one orientation, role map and location")
        index, swap, r, c = matches[0]
        difference = _transform(master != variant, index)
        h, w = difference.shape
        if h*w >= atlas.size:
            raise ValueError("reports must be proper crops")
        mask = np.zeros(atlas.shape, dtype=bool)
        mask[r:r+h, c:c+w] = difference
        multiplicity += mask
        # Delta values are measured in atlas coordinates after role registration.
        additions = int(np.count_nonzero(mask & (binary == 0)))
        removals = int(np.count_nonzero(mask & (binary == 1)))
        records.append(dict(mask=mask, orientation=index, swap=swap, offset=(r, c),
                            shape=(h, w), local_shape=patch.shape,
                            copies=len(panels), column=column,
                            additions=additions, removals=removals))
    info.update(binary=binary, records=records, multiplicity=multiplicity)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the visibly supplied atlas after all registered parity edits."""
    info = _analyze_input(input_grid)
    binary = info['binary'] ^ (info['multiplicity'] % 2).astype(np.int8)
    return np.where(binary, info['colors'][1], info['colors'][0]).astype(np.int8)


def _without_registration(info):
    """Counterfactual: put unrotated differences at the atlas origin."""
    wrong = np.zeros(info['atlas'].shape, dtype=np.int8)
    for panels, record in zip(info['packets'], info['records']):
        variant = panels[record['column']]
        master = next(p for j, p in enumerate(panels) if j != record['column'])
        h, w = master.shape
        if h > wrong.shape[0] or w > wrong.shape[1]:
            return None
        wrong[:h, :w] ^= (master != variant).astype(np.int8)
    return wrong


def _crop_bank(atlas):
    """Unique D4/complement classes of actual windows; no source-template bank."""
    classes = {}
    H, W = atlas.shape
    for h in range(3, min(6, H-1)+1):
        for w in range(3, min(6, W-1)+1):
            for r in range(H-h+1):
                for c in range(W-w+1):
                    patch = atlas[r:r+h, c:c+w]
                    if not 0 < patch.sum() < patch.size:
                        continue
                    orbit = {_key(t if swap == 0 else 1-t)
                             for i in range(8) for t in [_transform(patch, i)]
                             for swap in (0, 1)}
                    canonical = min(orbit)
                    if canonical in classes:
                        classes[canonical] = None
                    elif len(orbit) == 16:
                        classes[canonical] = (r, c, h, w)
                    else:
                        classes[canonical] = None
    return [v for v in classes.values() if v is not None]


def _draw_atlas(rng, H, W, style):
    a = np.zeros((H, W), dtype=np.int8)
    if style == 0:
        # Occupancy texture is one style; the other five use readable primitives.
        return (rng.random((H, W)) < rng.uniform(.3, .65)).astype(np.int8)
    for _ in range(int(rng.integers(3, 7))):
        kind = int(rng.integers(5)) if rng.random() < .45 else style-1
        r0, r1 = sorted(map(int, rng.choice(H, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(W, 2, replace=False)))
        if kind == 0:  # bent bars
            a[r0:r1+1, c0] = 1
            a[r1, c0:c1+1] = 1
            if rng.random() < .5:
                a[r0:r1+1, c1] = 1
        elif kind == 1:  # hollow rectangles
            a[r0:r1+1, c0] = a[r0:r1+1, c1] = 1
            a[r0, c0:c1+1] = a[r1, c0:c1+1] = 1
        elif kind == 2:  # staircase strokes
            length = max(r1-r0, c1-c0)+1
            rr = np.rint(np.linspace(r0, r1, length)).astype(int)
            cc = np.rint(np.linspace(c0, c1, length)).astype(int)
            a[rr, cc] = 1
            if rng.random() < .5:
                a[rr, np.maximum(0, cc-1)] = 1
        elif kind == 3:  # branched combs
            a[r0:r1+1, c0] = 1
            for r in range(r0, r1+1, int(rng.integers(2, 4))):
                a[r, c0:int(rng.integers(c0+1, c1+2))] = 1
        else:  # separated short bars / local repetition
            for r in range(r0, r1+1, 2):
                start = int(rng.integers(c0, c1+1))
                a[r, start:min(W, start+int(rng.integers(1, 4)))] = 1
    return a


def _points(window):
    r, c, h, w = window
    return {(y, x) for y in range(r, r+h) for x in range(c, c+w)}


def _choose_edits(rng, atlas, windows, witness):
    regions = [_points(w) for w in windows]
    if witness == 'parity':
        common = sorted(regions[0] & regions[1])
        rng.shuffle(common)
        for point in common:
            opposite = [[p for p in sorted(region) if atlas[p] != atlas[point]]
                        for region in regions]
            if opposite[0] and opposite[1]:
                first = opposite[0][int(rng.integers(len(opposite[0])))]
                second = [p for p in opposite[1] if p != first]
                if second:
                    return [{point, first}, {point, second[int(rng.integers(len(second)))]}]
        return None
    edits = []
    for i, region in enumerate(regions):
        points = sorted(region)
        if witness == 'directions':
            eligible = [p for p in points if atlas[p] == i]
            size = int(rng.integers(1, min(3, len(eligible))+1))
            chosen = rng.choice(len(eligible), size, replace=False)
            edits.append({eligible[int(j)] for j in chosen})
            continue
        mode = int(rng.integers(5))
        zeros = [p for p in points if atlas[p] == 0]
        ones = [p for p in points if atlas[p] == 1]
        if mode <= 1:
            eligible = zeros if mode == 0 else ones
            size = int(rng.integers(1, min(4, len(eligible))+1))
            edits.append({eligible[int(j)] for j in rng.choice(len(eligible), size, replace=False)})
        elif mode == 2:
            size = int(rng.integers(1, min(3, len(zeros), len(ones))+1))
            edits.append({zeros[int(j)] for j in rng.choice(len(zeros), size, replace=False)}
                         | {ones[int(j)] for j in rng.choice(len(ones), size, replace=False)})
        elif mode == 3:
            size = int(rng.integers(1, min(5, len(points))+1))
            edits.append({points[int(j)] for j in rng.choice(len(points), size, replace=False)})
        else:
            anchor = points[int(rng.integers(len(points)))]
            axis = int(rng.integers(2))
            eligible = [p for p in points if p[axis] == anchor[axis]]
            size = int(rng.integers(1, min(4, len(eligible))+1))
            edits.append({eligible[int(j)] for j in rng.choice(len(eligible), size, replace=False)})
    if len(regions) == 3 and rng.random() < .65:
        common = sorted(set.intersection(*regions))
        if common:
            point = common[int(rng.integers(len(common)))]
            for edit in edits:
                edit.add(point)
    return edits


def _frame_atlas(atlas, frame):
    box = np.full((atlas.shape[0]+2, atlas.shape[1]+2), frame, dtype=np.int8)
    box[1:-1, 1:-1] = atlas
    return box


def _frame_report(master, variant, copies, column, frame):
    h, w = master.shape
    box = np.full((h+2, copies*(w+1)+1), frame, dtype=np.int8)
    for j in range(copies):
        x = 1+j*(w+1)
        box[1:-1, x:x+w] = variant if j == column else master
    return box


def _pack(rng, atlas_box, reports, background, layout):
    reports = [reports[int(i)] for i in rng.permutation(len(reports))]
    if layout == 0:
        boxes = reports.copy()
        boxes.insert(int(rng.integers(len(boxes)+1)), atlas_box)
        H = sum(b.shape[0] for b in boxes)+len(boxes)-1
        W = max(b.shape[1] for b in boxes)
        H += int(rng.integers(min(3, 28-H)+1))
        W += int(rng.integers(min(3, 28-W)+1))
        result = np.full((H+2, W+2), background, dtype=np.int8)
        y = 1+int(rng.integers(H-(sum(b.shape[0] for b in boxes)+len(boxes)-1)+1))
        for box in boxes:
            x = 1+int(rng.integers(W-box.shape[1]+1))
            result[y:y+box.shape[0], x:x+box.shape[1]] = box
            y += box.shape[0]+1
        return result
    report_h = sum(b.shape[0] for b in reports)+len(reports)-1
    report_w = max(b.shape[1] for b in reports)
    H = max(atlas_box.shape[0], report_h)
    W = atlas_box.shape[1]+1+report_w
    H += int(rng.integers(min(3, 28-H)+1))
    W += int(rng.integers(min(3, 28-W)+1))
    result = np.full((H+2, W+2), background, dtype=np.int8)
    atlas_left = bool(rng.integers(2))
    ax = 1 if atlas_left else W-atlas_box.shape[1]+1
    ay = 1+int(rng.integers(H-atlas_box.shape[0]+1))
    result[ay:ay+atlas_box.shape[0], ax:ax+atlas_box.shape[1]] = atlas_box
    rx = atlas_box.shape[1]+2 if atlas_left else 1
    y = 1+int(rng.integers(H-report_h+1))
    for box in reports:
        x = rx+int(rng.integers(report_w-box.shape[1]+1))
        result[y:y+box.shape[0], x:x+box.shape[1]] = box
        y += box.shape[0]+1
    return result


def _make_pair(rng, palette, orientations, scene_orientation, witness, style):
    n = len(orientations)
    background, frame, low, high = palette
    for _ in range(64):
        layout = int(rng.integers(2))
        if layout == 0:
            maximum_h = {1: 16, 2: 11, 3: 8}[n]
            H, W = int(rng.integers(7, maximum_h+1)), int(rng.integers(7, 17))
        else:
            H, W = int(rng.integers(7, 17)), int(rng.integers(7, 11))
        atlas = _draw_atlas(rng, H, W, style)
        if not .15 < atlas.mean() < .8:
            continue
        bank = _crop_bank(atlas)
        if len(bank) < 8:
            continue
        options = []
        for orientation in orientations:
            available = []
            for window in bank:
                r, c, h, w = window
                ph, pw = _transform(atlas[r:r+h, c:c+w], orientation).shape
                width_budget = 28 if layout == 0 else 25-W
                max_copies = min(6, (width_budget-1)//(pw+1))
                if max_copies >= 3:
                    available.append((window, ph, pw, max_copies))
            options.append(available)
        if not all(options):
            continue
        for _ in range(120):
            selected = [items[int(rng.integers(len(items)))] for items in options]
            windows = [item[0] for item in selected]
            if witness and windows[0] == windows[1]:
                continue
            copies = [int(rng.integers(3, item[3]+1)) for item in selected]
            report_h = sum(item[1]+2 for item in selected)+n-1
            report_w = max(k*(item[2]+1)+1 for k, item in zip(copies, selected))
            if ((layout == 0 and H+2+report_h+1 > 28)
                    or (layout == 1 and max(H+2, report_h) > 28)):
                continue
            edits = _choose_edits(rng, atlas, windows, witness)
            if edits is None:
                continue
            multiplicity = np.zeros(atlas.shape, dtype=int)
            for edit in edits:
                for point in edit:
                    multiplicity[point] += 1
            if not np.any(multiplicity % 2):
                continue
            reports = []
            columns = []
            for i, (window, orientation, count, edit) in enumerate(zip(windows, orientations, copies, edits)):
                r, c, h, w = window
                patch = atlas[r:r+h, c:c+w].copy()
                variant = patch.copy()
                for y, x in edit:
                    variant[y-r, x-c] ^= 1
                swap = i % 2 if witness else int(rng.integers(2))
                patch, variant = _transform(patch, orientation), _transform(variant, orientation)
                patch, variant = patch ^ swap, variant ^ swap
                column_choices = [j for j in range(count)
                                  if not (witness == 'directions' and i == 1 and j == columns[0])]
                column = column_choices[int(rng.integers(len(column_choices)))]
                columns.append(column)
                patch = np.where(patch, high, low).astype(np.int8)
                variant = np.where(variant, high, low).astype(np.int8)
                reports.append(_frame_report(patch, variant, count, column, frame))
            colored_atlas = np.where(atlas, high, low).astype(np.int8)
            inp = _pack(rng, _frame_atlas(colored_atlas, frame), reports, background, layout)
            binary_out = atlas ^ (multiplicity % 2).astype(np.int8)
            out = np.where(binary_out, high, low).astype(np.int8)
            return {'input': _transform(inp, scene_orientation),
                    'output': _transform(out, scene_orientation)}
    raise RuntimeError("bounded atlas/crop construction exhausted")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    shared_palette = rng.random() < .25
    style = int(rng.integers(6))
    quarter = int(rng.choice([1, 3]))
    reflection = int(rng.integers(4, 8))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for j in range(count):
            witness = 'directions' if split == 'train' and j == 0 else 'parity' if split == 'train' and j == 1 else None
            if witness == 'directions':
                orientations, scene = [0, quarter], 0
            elif witness == 'parity':
                orientations, scene = [reflection, int(rng.choice([0, quarter]))], 1
            else:
                n = int(rng.choice([1, 2, 3], p=[.18, .4, .42]))
                pool = [0, 1, 3, 4, 5, 6, 7] if split == 'train' else list(range(8))
                orientations = list(map(int, rng.choice(pool, n)))
                if split == 'test':
                    orientations[0] = 2  # Held-out half-turn; invariant under scene conjugation.
                scene = int(rng.integers(8))
            for _ in range(16):
                if not shared_palette:
                    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _make_pair(rng, palette, orientations, scene, witness,
                                  style if rng.random() < .6 else int(rng.integers(6)))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate input retry exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze_input(inp)
            assert len(np.unique(inp)) == 4
            assert out.shape == info['atlas'].shape
            assert np.array_equal(solve(inp), out)
            parity = info['multiplicity'] % 2 != 0
            assert np.any(parity), 'generated examples must demonstrate a real edit'
            assert np.array_equal(out != info['atlas'], parity)
            assert np.array_equal(out[~parity], info['atlas'][~parity])
            assert set(map(int, np.unique(out))) <= set(info['colors'])
            features[split].append(info)
    training = features['train']
    records = [r for info in training for r in info['records']]
    assert any(r['additions'] > 0 and r['removals'] == 0 for r in records)
    assert any(r['removals'] > 0 and r['additions'] == 0 for r in records)
    assert any(r['additions'] == r['removals'] > 0 for r in records)
    assert any(np.any((info['multiplicity'] > 0) & (info['multiplicity'] % 2 == 0))
               and np.any(info['multiplicity'] % 2) for info in training), 'show cancellation AND surviving edits'
    assert any(len({r['column'] for r in info['records']}) > 1 for info in training)
    assert any(_without_registration(info) is None
               or not np.array_equal(_without_registration(info), info['multiplicity'] % 2)
               for info in training), 'omitting crop registration must change a demonstrated answer'
    assert {0} < {r['orientation'] for r in records}, 'registration cannot ignore orientation'
    assert {r['swap'] for r in records} == {0, 1}
    assert all(r['orientation'] != 2 for r in records)
    assert all(any(r['orientation'] == 2 for r in info['records']) for info in features['test'])


def _signature(p):
    return tuple((s, _key(x['input']), _key(x['output'])) for s in ('train', 'test') for x in p[s])


def self_test():
    """Bounded 42-seed variation, semantic invariants and corruption checks."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in corpus}) == 42
    assert _signature(corpus[0]) == _signature(generate_puzzle(0))
    infos = [_analyze_input(pair['input']) for p in corpus for s in ('train', 'test') for pair in p[s]]
    records = [r for info in infos for r in info['records']]
    atlas_classes = {min(_key(_transform(info['binary'], t) ^ swap)
                         for t in range(8) for swap in (0, 1)) for info in infos}
    assert len(atlas_classes) >= 100
    assert len({info['atlas'].shape for info in infos}) >= 25
    assert {len(info['records']) for info in infos} == {1, 2, 3}
    assert len({r['local_shape'] for r in records}) >= 12
    assert len({r['offset'] for r in records}) >= 35
    assert {r['orientation'] for r in records} == set(range(8))
    assert len({r['copies'] for r in records}) >= 3
    assert {info['background'] for info in infos} == set(range(10))
    assert {info['frame'] for info in infos} == set(range(10))
    assert any(r['additions']+r['removals'] >= 5 for r in records)
    assert any(len({r['column'] for r in info['records']}) < len(info['records']) for info in infos)
    assert any(np.any(info['multiplicity'] >= 3) for info in infos)
    # Color permutation must commute with solving, including marker roles.
    permutation = np.array([9, 4, 7, 1, 6, 0, 8, 2, 5, 3], dtype=np.int8)
    pair = corpus[0]['train'][0]
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Rotation/reflection of the whole scene rotates/refects its answer.
    for t in range(8):
        assert np.array_equal(solve(_transform(pair['input'], t)), _transform(pair['output'], t))
    # Duplicate reports are separate toggle instructions. Two cancel; three survive.
    info = _analyze_input(pair['input'])
    r, c, h, w = _crop_bank(info['binary'])[0]
    master = info['atlas'][r:r+h, c:c+w].copy()
    variant = master.copy()
    variant[0, 0] = next(v for v in info['colors'] if v != int(variant[0, 0]))
    report = _frame_report(master, variant, 3, 1, info['frame'])
    for count in (2, 3):
        inp = _pack(np.random.default_rng(count), _frame_atlas(info['atlas'], info['frame']),
                    [report]*count, info['background'], 0)
        expected = info['atlas'].copy()
        if count == 3:
            expected[r, c] = variant[0, 0]
        assert np.array_equal(solve(inp), expected)
    absent = _frame_report(master, master, 3, 1, info['frame'])
    invalid = _pack(np.random.default_rng(9), _frame_atlas(info['atlas'], info['frame']),
                    [absent], info['background'], 0)
    broken = pair['input'].copy()
    y, x = np.argwhere(broken == info['frame'])[0]
    broken[y, x] = info['background']
    ambiguous_atlas = np.indices((8, 8)).sum(axis=0) % 2
    ambiguous_atlas = np.where(ambiguous_atlas, info['colors'][1], info['colors'][0]).astype(np.int8)
    amb_master = ambiguous_atlas[:3, :3].copy()
    amb_variant = amb_master.copy()
    amb_variant[1, 1] = next(v for v in info['colors'] if v != int(amb_variant[1, 1]))
    ambiguous = _pack(np.random.default_rng(10), _frame_atlas(ambiguous_atlas, info['frame']),
                      [_frame_report(amb_master, amb_variant, 3, 1, info['frame'])], info['background'], 0)
    for inp in (invalid, broken, ambiguous):
        try:
            solve(inp)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed, absent-exception or ambiguous report accepted')
    for nt, nq in ((2, 1), (10, 4)):
        assert len(generate_puzzle(120+nt, nt, nq)['train']) == nt
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in corpus[0][s]] for s in corpus[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('altered output accepted')
