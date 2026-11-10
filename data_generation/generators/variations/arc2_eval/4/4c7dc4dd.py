"""Camouflaged, locally demonstrated visual analogies.

Member order carries no semantics. A complete pair supplies the operation or
blueprint; the occupied target supplies its state and a visible color map.
"""
import hashlib
from itertools import combinations, permutations

import numpy as np


def _as_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    a, b = _as_count(num_train, "num_train"), _as_count(num_test, "num_test")
    if a is not None and b is not None and a <= b:
        raise ValueError("num_train must exceed num_test")
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1 if a is not None else 5))
    if a is None:
        a = int(rng.integers(max(2, b + 1), 11))
    return a, b


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _outline(grid, box, color):
    r, c, h, w = box
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _inside(a, b):
    """Rectangle a lies strictly inside b."""
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r+h <= y or y+bh <= r or c+w <= x or x+bw <= c


def _rectangles(grid):
    """Find monochrome outlines, without assumptions about their colors/gaps."""
    h, w = grid.shape
    found = []
    for color in np.unique(grid):
        mask = grid == color
        prefix = np.vstack((np.zeros((1, w), dtype=int), np.cumsum(mask, axis=0)))
        for top in range(h - 5):
            for bottom in range(top + 5, h):
                vertical = np.flatnonzero(prefix[bottom+1] - prefix[top] == bottom-top+1)
                if len(vertical) < 2:
                    continue
                edge = mask[top] & mask[bottom]
                bad = np.r_[0, np.cumsum(~edge)]
                for left, right in combinations(vertical.tolist(), 2):
                    if right-left >= 5 and bad[right+1] == bad[left]:
                        found.append((top, left, bottom-top+1, right-left+1, int(color)))
    return found


def _extract_scene(grid):
    rectangles = _rectangles(grid)
    small = [b for b in rectangles if 6 <= b[2] <= 8 and 6 <= b[3] <= 8
             and not np.all(grid[b[0]+1:b[0]+b[2]-1, b[1]+1:b[1]+b[3]-1] == b[4])]
    # Motif outlines inside actual local frames are not additional panels.
    small = [a for a in small if not any(_inside(a, b) for b in small)]
    groups = {}
    for outer in rectangles:
        members = tuple(i for i, panel in enumerate(small) if _inside(panel, outer))
        if len(members) != 2:
            continue
        a, b = [small[i] for i in members]
        if a[2:4] != b[2:4] or not _disjoint(a, b):
            continue
        groups.setdefault(members, []).append(outer)
    scenes = {}
    for (first, boxes1), (second, boxes2) in combinations(groups.items(), 2):
        if set(first) & set(second):
            continue
        members = first + second
        if len({small[i][2:4] for i in members}) != 1:
            continue
        valid_boxes = [(a, b) for a in boxes1 for b in boxes2 if _disjoint(a, b)]
        if not valid_boxes:
            continue
        boxes = min(valid_boxes, key=lambda ab: sum(x[2]*x[3] for x in ab))
        clusters = [[small[i] for i in indices] for indices in (first, second)]
        if boxes[0][:2] > boxes[1][:2]:
            boxes, clusters = boxes[::-1], clusters[::-1]
        for cluster in clusters:
            a, b = cluster
            horizontal = a[1]+a[3] <= b[1] or b[1]+b[3] <= a[1]
            cluster.sort(key=lambda p: p[1] if horizontal else p[0])
        interiors = [[grid[r+1:r+h-1, c+1:c+w-1].copy() for r,c,h,w,_ in cluster]
                     for cluster in clusters]
        blanks = [(g, s, int(p[0, 0])) for g in range(2) for s,p in enumerate(interiors[g])
                  if np.all(p == p[0, 0])]
        if len(blanks) != 1:
            continue
        target, missing, neutral = blanks[0]
        if any(p[4] == neutral for cluster in clusters for p in cluster):
            continue
        signature = tuple(sorted(tuple(small[i]) for i in members)), first, second
        scenes[signature] = dict(reference=interiors[1-target], occupied=interiors[target][1-missing],
                                 neutral=neutral, boxes=boxes, panels=clusters,
                                 reference_group=1-target, occupied_slot=1-missing,
                                 shape=interiors[0][0].shape)
    assert len(scenes) == 1, f"expected a unique four-panel association, got {len(scenes)}"
    return next(iter(scenes.values()))


def _fill_cues(grid, neutral, primary, axes):
    """Simultaneous closure of original aligned cues; never grow from new cells."""
    mask = grid != neutral
    additions = np.zeros(grid.shape, dtype=bool)
    for axis in axes:
        rows = mask if axis == 0 else mask.T
        dest = additions if axis == 0 else additions.T
        for row, occupied in enumerate(rows):
            positions = np.flatnonzero(occupied)
            if len(positions) >= 2:
                dest[row, positions[0]:positions[-1]+1] = True
    out = grid.copy()
    out[additions & ~mask] = primary
    return out


def _endpoints(grid, neutral, axis):
    src = grid if axis == 0 else grid.T
    out = np.full_like(src, neutral)
    for row, values in enumerate(src):
        positions = np.flatnonzero(values != neutral)
        if len(positions):
            out[row, positions[[0, -1]]] = values[positions[[0, -1]]]
    return out if axis == 0 else out.T


def _maps(source, target, neutral):
    a = sorted(set(map(int, np.unique(source))) - {neutral})
    b = sorted(set(map(int, np.unique(target))) - {neutral})
    if len(a) != len(b) or not a:
        return []
    result = []
    for ordered in permutations(b):
        mapping = dict(zip(a, ordered))
        # Shared visible roles are retained. Disjoint palettes use motif evidence.
        if any(mapping[x] != x for x in set(a) & set(b)):
            continue
        result.append({neutral: neutral, **mapping})
    return result


def _recolor(grid, mapping):
    if not set(map(int, np.unique(grid))) <= set(mapping):
        return None
    return np.array([[mapping[int(x)] for x in row] for row in grid], dtype=np.int8)


def _prediction_candidates(reference, occupied, neutral):
    predictions = {}
    def add(answer, family):
        if answer is not None and not np.all(answer == neutral):
            predictions.setdefault(_key(answer), (answer, set()))[1].add(family)

    for source, partner in (reference, reference[::-1]):
        source_colors = sorted(set(map(int, np.unique(source))) - {neutral})
        for mapping in _maps(source, occupied, neutral):
            transferred = _recolor(partner, mapping)
            # A complete blueprint can be transferred only when the target
            # actually matches a reference state, including every color role.
            for flip in (0, 1, 2, 3):
                state = source
                other = partner
                if flip & 1:
                    state, other = state[::-1], other[::-1]
                if flip & 2:
                    state, other = state[:, ::-1], other[:, ::-1]
                if flip and not np.array_equal(other, partner):
                    continue  # no unsupported movement of an asymmetric answer
                if np.array_equal(_recolor(state, mapping), occupied):
                    add(transferred, "blueprint")
            if len(source_colors) == 1:
                primary = source_colors[0]
                if np.array_equal(partner, np.where(source == neutral, primary, neutral)):
                    add(np.where(occupied == neutral, mapping[primary], neutral).astype(np.int8), "complement")
            for primary in source_colors:
                # When both colors change, a single accent cue identifies the
                # exceptional role; otherwise geometry must establish the map.
                if len(source_colors) == 2 and not (set(source_colors) & set(mapping.values())):
                    accent = next(c for c in source_colors if c != primary)
                    if np.count_nonzero(source == accent) != 1 or np.count_nonzero(occupied == mapping[accent]) != 1:
                        continue
                for axes in ((0,), (1,), (0, 1)):
                    filled = _fill_cues(source, neutral, primary, axes)
                    if not np.array_equal(source, filled) and np.array_equal(filled, partner):
                        answer = _fill_cues(occupied, neutral, mapping[primary], axes)
                        if not np.array_equal(answer, occupied):
                            add(answer, "line")
                for axis in (0, 1):
                    ends = _endpoints(source, neutral, axis)
                    if (not np.array_equal(ends, source) and np.array_equal(ends, partner)
                            and np.array_equal(_fill_cues(partner, neutral, primary, (axis,)), source)):
                        answer = _endpoints(occupied, neutral, axis)
                        if (not np.array_equal(answer, occupied)
                                and np.array_equal(_fill_cues(answer, neutral, mapping[primary], (axis,)), occupied)):
                            add(answer, "endpoints")
    return predictions


def _solve_scene(scene):
    assert _reference_is_supported(scene["reference"], scene["neutral"]), "unsupported reference motif"
    predictions = _prediction_candidates(scene["reference"], scene["occupied"], scene["neutral"])
    assert len(predictions) == 1, f"reference permits {len(predictions)} distinct predictions"
    return next(iter(predictions.values()))[0].copy()


def _reference_is_supported(reference, neutral):
    colors = set(map(int,np.unique(np.stack(reference)))) - {neutral}
    if not 1 <= len(colors) <= 2:
        return False
    if _distance_features(reference, neutral) is not None:
        return True
    for a,b in (reference, reference[::-1]):
        if len(colors) == 1 and np.array_equal(b,np.where(a == neutral,next(iter(colors)),neutral)):
            return True
        for primary in colors:
            for axes in ((0,), (1,), (0,1)):
                if not np.array_equal(a,b) and np.array_equal(_fill_cues(a,neutral,primary,axes),b):
                    return True
        if (np.array_equal(b,b[::-1]) and np.array_equal(b,b[:,::-1])
                and np.all((a == neutral) | (a == b))
                and 3 <= np.count_nonzero(a != neutral) < np.count_nonzero(b != neutral)
                and set(map(int,np.unique(a))) - {neutral} == colors):
            return True
    return False


def solve(train_pairs, test_input):
    """Data-only prediction check; no outputs or generation state for the test."""
    for pair in train_pairs:
        assert np.array_equal(_solve_scene(_extract_scene(pair["input"])), pair["output"])
    return _solve_scene(_extract_scene(test_input))


def _line_motif(rng, shape, axes, mode):
    h, w = shape
    for _ in range(80):
        a = np.zeros(shape, dtype=np.int8)
        if mode == 0:
            axis = axes[0]
            row = int(rng.integers(h if axis == 0 else w))
            length = w if axis == 0 else h
            lo = int(rng.integers(length-2))
            hi = int(rng.integers(lo+2, length))
            points = [(row, lo), (row, hi)] if axis == 0 else [(lo, row), (hi, row)]
        else:
            count = int(rng.integers(3, min(8, h*w//2)))
            points = [divmod(int(i), w) for i in rng.choice(h*w, count, replace=False)]
        for p in points:
            a[p] = 1
        a[points[int(rng.integers(len(points)))]] = 2
        b = _fill_cues(a, 0, 1, axes)
        if np.count_nonzero(b != a) >= 1:
            return a, b
    # Deterministic, valid simple end of the family.
    a = np.zeros(shape, dtype=np.int8)
    a[0, 0], a[0, -1] = 2, 1
    return a, _fill_cues(a, 0, 1, (0,))


def _binary_motif(rng, shape):
    h, w = shape
    rr, cc = np.indices(shape)
    mode = int(rng.integers(6))
    if mode == 0:
        mask = rng.random(shape) < rng.uniform(.12, .88)
    elif mode == 1:
        mask = np.zeros(shape, bool)
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(3, h*w*2))):
            mask[r, c] = True
            dr, dc = ((0,1),(0,-1),(1,0),(-1,0))[int(rng.integers(4))]
            r, c = np.clip(r+dr, 0, h-1), np.clip(c+dc, 0, w-1)
    elif mode == 2:
        r0, r1 = sorted(rng.choice(h+1, 2, replace=False))
        c0, c1 = sorted(rng.choice(w+1, 2, replace=False))
        mask = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    elif mode == 3:
        mask = (rr + int(rng.integers(3))*cc) % int(rng.integers(2, 4)) == 0
    elif mode == 4:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    else:
        mask = abs(rr-int(rng.integers(h))) + abs(cc-int(rng.integers(w))) < int(rng.integers(2, max(h,w)))
    if rng.integers(2):
        mask = ~mask
    if mask.all() or not mask.any():
        mask[0, 0], mask[-1, -1] = False, True
    return mask.astype(np.int8), (~mask).astype(np.int8)


def _distance(rng, shape):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(40):
        if rng.random() < .55:
            rows, cols = ((h-1)//2, h//2), ((w-1)//2, w//2)
        else:
            r, c = int(rng.integers(h)), int(rng.integers(w))
            rows, cols = (r, r), (c, c)
        d = np.minimum(abs(rr-rows[0]), abs(rr-rows[1])) + np.minimum(abs(cc-cols[0]), abs(cc-cols[1]))
        maximum = int(d.max())
        if maximum < 3:
            continue
        extent = int(rng.integers(2, min(4, maximum)))
        radius = int(rng.integers(extent, maximum))
        colors = 1 + (d + int(rng.integers(2))) % 2
        a, b = np.where(d < extent, colors, 0), np.where(d >= radius, colors, 0)
        if all(set(np.unique(x)) == {0, 1, 2} for x in (a, b)):
            return a.astype(np.int8), b.astype(np.int8)
    d = abs(rr-h//2) + abs(cc-w//2)
    return np.where(d < 2, 1+d%2, 0).astype(np.int8), np.where(d >= 2, 1+d%2, 0).astype(np.int8)


def _symmetry(rng, shape, endpoints=False):
    h, w = shape
    half = (h+1)//2
    for _ in range(60):
        # A procedural symmetric silhouette, often a stepped filled figure.
        widths = rng.integers(1, (w+1)//2+1, half)
        if rng.random() < .65:
            widths.sort()
        widths = np.r_[widths, widths[-(2 if h%2 else 1)::-1]]
        b = np.zeros(shape, dtype=np.int8)
        for r, radius in enumerate(widths):
            left = max(0, w//2-int(radius)) if w%2 == 0 else max(0, w//2-int(radius)+1)
            b[r, left:w-left] = 1
            if rng.random() < .55 and r < half:
                b[r, left] = b[r, w-1-left] = 2
        b[half:] = b[:h//2][::-1]
        if len(np.unique(b)) < 3:
            continue
        if endpoints:
            a = _endpoints(b, 0, 0)
        else:
            a = np.zeros_like(b)
            positions = np.argwhere((b != 0) & (np.indices(shape)[0] < half))
            corner = (int(rng.choice([0, h-1])), int(rng.choice([0, w-1])))
            positions = sorted(map(tuple, positions), key=lambda p: (abs(p[0]-corner[0])+abs(p[1]-corner[1]), p))
            # Preserve both roles in a small connected corner/edge cue.
            for p in positions[:int(rng.integers(3, min(7, len(positions))+1))]:
                a[p] = b[p]
            if rng.integers(2):
                a = a[::-1].copy()
            if rng.integers(2):
                a = a[:, ::-1].copy()
        if set(np.unique(a)) == {0, 1, 2} and np.count_nonzero(b) > np.count_nonzero(a):
            if endpoints and rng.integers(2):
                return a.T.copy(), b.T.copy()
            return a, b
    a = np.zeros(shape, dtype=np.int8)
    a[0, :3] = [2, 1, 1]
    b = np.ones(shape, dtype=np.int8)
    b[0, 0] = b[-1, 0] = b[0, -1] = b[-1, -1] = 2
    return a, b


def _relation(rng, kind, shape):
    if kind == "line":
        axes = ((0,), (1,), (0,1))[int(rng.integers(3))]
        mode = int(rng.choice([0, 1, 1, 1]))
        a, b = _line_motif(rng, shape, axes, mode)
        t, u = _line_motif(rng, shape, axes, mode)
        if mode == 0 and rng.random() < .3:
            t, u = u, t
    elif kind == "complement":
        a, b = _binary_motif(rng, shape)
        t, u = _binary_motif(rng, shape)
    elif kind == "distance":
        a, b = _distance(rng, shape)
        t, u = (a.copy(), b.copy()) if rng.integers(2) else (b.copy(), a.copy())
    else:
        endpoints = rng.random() < .3
        a, b = _symmetry(rng, shape, endpoints)
        t, u = a.copy(), b.copy()
        if not endpoints:
            if rng.integers(2):
                t = t[::-1].copy()
            if rng.integers(2):
                t = t[:, ::-1].copy()
        elif rng.random() < .25:
            t, u = u, t
    return a, b, t, u


def _layout(rng, shape, mixed, spacious):
    ph, pw = shape[0]+2, shape[1]+2
    orientations = [0, 1] if mixed else [0, 0]
    rng.shuffle(orientations)
    minima = [(ph+2, 2*pw+3) if axis == 0 else (2*ph+3, pw+2) for axis in orientations]
    min_h, min_w = sum(x[0] for x in minima)+1, max(x[1] for x in minima)
    if min_h > 30:
        return _layout(rng, shape, False, spacious)
    height = 30 if spacious else int(rng.integers(min_h, 31))
    width = 30 if spacious else int(rng.integers(min_w, 31))
    slack = height-min_h
    allocation = rng.multinomial(slack, [1/5]*5)
    heights = [minima[0][0]+int(allocation[0]), minima[1][0]+int(allocation[1])]
    tops = [int(allocation[2]), int(allocation[2])+heights[0]+1+int(allocation[3])]
    boxes, panels = [], []
    for i, axis in enumerate(orientations):
        gh = heights[i]
        gw = int(rng.integers(minima[i][1], width+1))
        r, c = tops[i], int(rng.integers(width-gw+1))
        boxes.append((r, c, gh, gw))
        span, size = (gw, pw) if axis == 0 else (gh, ph)
        gap = int(rng.integers(1, span-2*size-1))
        start = int(rng.integers(1, span-2*size-gap))
        along = [start, start+size+gap]
        if axis == 0:
            positions = [(r+int(rng.integers(1, gh-ph)), c+x) for x in along]
        else:
            positions = [(r+x, c+int(rng.integers(1, gw-pw))) for x in along]
        panels.append(positions)
    return height, width, boxes, panels


def _wallpaper(rng, height, width, palette):
    rr, cc = np.indices((height, width))
    mode = int(rng.integers(3))
    if mode == 0:
        indices = (rr + cc*int(rng.integers(1, len(palette))) + int(rng.integers(10))) % len(palette)
        grid = palette[indices]
    elif mode == 1:
        tile = rng.choice(palette, size=(int(rng.integers(2, 6)), int(rng.integers(2, 6))))
        grid = tile[rr % tile.shape[0], cc % tile.shape[1]]
    else:
        grid = palette[(rr//int(rng.integers(1, 3)) + cc//int(rng.integers(1, 3))) % len(palette)]
    noise = rng.random(grid.shape) < rng.uniform(.03, .18)
    grid[noise] = rng.choice(palette, np.count_nonzero(noise))
    # Short repeats are camouflage; prevent complete accidental panel outlines.
    for r in range(height):
        for c in range(width):
            forbidden = set()
            if c >= 3 and grid[r,c-1] == grid[r,c-2] == grid[r,c-3]:
                forbidden.add(int(grid[r,c-1]))
            if r >= 3 and grid[r-1,c] == grid[r-2,c] == grid[r-3,c]:
                forbidden.add(int(grid[r-1,c]))
            if int(grid[r,c]) in forbidden:
                grid[r,c] = next(int(x) for x in palette if int(x) not in forbidden)
    return grid.astype(np.int8)


def _make_pair(rng, kind, style, split):
    for attempt in range(120):
        side = int(rng.choice([4,5,6]))
        shape = (side, int(rng.choice([4,5,6]))) if kind in ("line", "complement") and rng.random() < .35 else (side, side)
        logical = _relation(rng, kind, shape)
        shape = logical[0].shape
        colors = style["colors"].copy() if rng.random() > style["variation"] else rng.permutation(10)
        neutral, primary, accent = map(int, colors[:3])
        mapping_mode = int(rng.integers(3))
        ref_map = {0:neutral, 1:primary, 2:accent}
        target_map = {0:neutral, 1:primary if mapping_mode == 0 else int(colors[3]),
                      2:accent if mapping_mode < 2 else int(colors[4])}
        a, b = [_recolor(x, ref_map) for x in logical[:2]]
        t, output = [_recolor(x, target_map) for x in logical[2:]]
        predictions = _prediction_candidates([a,b], t, neutral)
        if len(predictions) != 1 or not np.array_equal(next(iter(predictions.values()))[0], output):
            continue
        mixed = rng.random() < .17*style["variation"]
        height, width, boxes, positions = _layout(rng, shape, mixed, rng.random() < .25)
        wallpaper_count = int(rng.integers(3, 9 if split == "target_test" else 8))
        palette = rng.choice(colors, wallpaper_count, replace=False)
        grid = _wallpaper(rng, height, width, palette)
        group_colors = [int(colors[5]), int(colors[5] if rng.random() < .3 else colors[6])]
        frame_colors = [int(colors[7]), int(colors[7] if rng.random() < .45 else colors[8])]
        ref_group = int(rng.integers(2))
        if rng.random() < .18:
            group_colors[ref_group] = primary
        if rng.random() < .18:
            group_colors[1-ref_group] = target_map[1]
        if rng.random() < .4:
            frame_colors[ref_group] = primary
        if rng.random() < .4:
            frame_colors[1-ref_group] = target_map[1]
        reference = [a,b] if rng.integers(2) else [b,a]
        target = [t, np.full(shape, neutral, dtype=np.int8)]
        if rng.integers(2):
            target.reverse()
        contents = [None, None]
        contents[ref_group], contents[1-ref_group] = reference, target
        for i, box in enumerate(boxes):
            _outline(grid, box, group_colors[i])
            for j, (r,c) in enumerate(positions[i]):
                _outline(grid, (r,c,shape[0]+2,shape[1]+2), frame_colors[i])
                grid[r+1:r+1+shape[0], c+1:c+1+shape[1]] = contents[i][j]
        transpose = style["transpose"] if rng.random() > style["variation"] else bool(rng.integers(2))
        if transpose:
            grid, output = grid.T.copy(), output.T.copy()
        try:
            scene = _extract_scene(grid)
            if not np.array_equal(_solve_scene(scene), output):
                continue
        except AssertionError:
            continue
        if _source_copy(grid):
            continue
        return {"input":grid, "output":output.copy()}
    return _fallback_pair(rng, kind, style)


def _fallback_pair(rng, kind, style):
    """A bounded, readable construction for each family if rejection exhausts."""
    rr, cc = np.indices((6,6))
    if kind == "line":
        a = np.zeros((6,6), dtype=np.int8)
        a[1,1], a[1,5] = 2, 1
        b = _fill_cues(a, 0, 1, (0,))
    elif kind == "complement":
        a = ((rr == 2) | (cc == 3)).astype(np.int8)
        b = 1-a
    elif kind == "distance":
        d = abs(rr-2)+abs(cc-2)
        a = np.where(d < 2, 1+d%2, 0).astype(np.int8)
        b = np.where(d >= 3, 1+d%2, 0).astype(np.int8)
    else:
        b = np.ones((6,6), dtype=np.int8)
        b[0,0] = b[0,-1] = b[-1,0] = b[-1,-1] = 0
        b[0,1] = b[0,-2] = b[-1,1] = b[-1,-2] = 2
        a = np.zeros_like(b)
        a[0,1:3], a[1,0] = b[0,1:3], b[1,0]
    grid = (8+(np.indices((30,30)).sum(axis=0) % 2)).astype(np.int8)
    boxes = [(1,1,12,28), (16,2,13,27)]
    panels = [[(3,3), (3,18)], [(18,4), (19,19)]]
    contents = [[a,b], [a,np.zeros_like(a)]]
    for i, box in enumerate(boxes):
        _outline(grid, box, 5+i)
        for (r,c), interior in zip(panels[i], contents[i]):
            _outline(grid, (r,c,8,8), 7)
            grid[r+1:r+7,c+1:c+7] = interior
    palette = style["colors"]
    grid, output = palette[grid].astype(np.int8), palette[b].astype(np.int8)
    if rng.integers(2):
        grid, output = grid.T.copy(), output.T.copy()
    assert np.array_equal(_solve_scene(_extract_scene(grid)), output)
    return {"input":grid, "output":output}


def _schedule(rng, a, b):
    families = ["line", "complement", "distance", "symmetry"]
    regime = int(rng.integers(4))
    if regime == 0:
        train_pool, test_pool = families[:2], families[2:]
    elif regime == 1:
        train_pool, test_pool = families[2:], families[:2]
    elif regime == 2:
        train_pool = test_pool = families
    else:
        train_pool = test_pool = [families[int(i)] for i in rng.choice(4, 2, replace=False)]
    train = [train_pool[i % len(train_pool)] for i in range(a)]
    test = [str(rng.choice(test_pool)) for _ in range(b)]
    rng.shuffle(train)
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    a, b = _resolve_counts(rng, num_train, num_test)
    # Only distributional preferences are seed-wide. The invariant convention
    # is local analogy with interior-only output, demonstrated by every train.
    style = dict(colors=rng.permutation(10), variation=float(rng.choice([0., .45, 1.])),
                 transpose=bool(rng.integers(2)))
    train, test = _schedule(rng, a, b)
    puzzle = {"train":[], "test":[]}
    seen = set()
    for split, kinds in (("train", train), ("test", test)):
        for kind in kinds:
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            for _ in range(20):
                pair = _make_pair(pair_rng, kind, style, "source_train" if split == "train" else "target_test")
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("could not construct distinct inputs")
    return puzzle


# Exact source input content fingerprints (uint8, row-major), not size bans.
_SOURCE_INPUT_HASHES = frozenset(['2d697af3b7644d08960bc376cc9624c2e0227baa2e1bf780493917f57be452ee', '472ced0e4cc910b4d891a800eae1fd912e6c70e7b16fa845223d1c3a50b86cec', 'bef2b735474ae621ea786d87732f9f3cab9304b6c4ab747f4749868de4589742', 'd859d660c36848bd6efb00490be9f5272b585ce51eca73664fb57622e4cf2093'])


def _source_copy(grid):
    return grid.shape == (30,30) and hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUT_HASHES


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            ik, ok = _key(pair["input"]), _key(pair["output"])
            assert ik not in seen_inputs and (ik,ok) not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add((ik,ok))
            assert not _source_copy(pair["input"])
            scene = _extract_scene(pair["input"])
            assert pair["output"].shape == scene["shape"]
            assert np.array_equal(_solve_scene(scene), pair["output"])
    # There are no unseen palette/position conventions: every demonstration
    # teaches the same locally supported reference-to-target reconstruction.


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x,y in zip(a[s], b[s])
                                           for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    shapes, layouts, palettes, masks, counts, tests = set(), set(), set(), set(), set(), set()
    review_records = []
    full_size = mixed = remapped = reused = varying_neutral = repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        neutrals = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                scene = _extract_scene(grid)
                review_records.append((seed, split, review_features(pair)))
                neutrals.add(scene["neutral"])
                shapes.add(grid.shape)
                full_size += grid.shape == (30,30)
                layouts.add(tuple(tuple(p) for group in scene["panels"] for p in group))
                palettes.add(tuple(map(int, np.unique(grid))))
                masks.add(_key((pair["output"] != scene["neutral"]).astype(np.int8)))
                ref = set(map(int,np.unique(scene["reference"][0]))) - {scene["neutral"]}
                target = set(map(int,np.unique(scene["occupied"]))) - {scene["neutral"]}
                remapped += not bool(ref & target)
                reused += any(p[4] in ref | target for group in scene["panels"] for p in group)
                axes = []
                for group in scene["panels"]:
                    x,y = group
                    axes.append(x[1]+x[3] <= y[1] or y[1]+y[3] <= x[1])
                mixed += axes[0] != axes[1]
                outside = np.ones(grid.shape, bool)
                for r,c,h,w,_ in scene["boxes"]:
                    outside[r:r+h,c:c+w] = False
                repeated += bool(np.any((grid[:,:-1] == grid[:,1:]) & outside[:,:-1] & outside[:,1:]))
        varying_neutral += len(neutrals) > 1
    assert len(shapes) >= 30 and len(layouts) >= 200 and len(palettes) >= 25 and len(masks) >= 100
    assert len(counts) >= 4 and len(tests) >= 3
    assert min(full_size, mixed, remapped, reused, varying_neutral, repeated) > 0
    _review_self_test(review_records)
    for i, (a,b) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        puzzle = generate_puzzle(1200+i, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        assert _puzzles_equal(puzzle, generate_puzzle(1200+i, a, b))
    assert _puzzles_equal(generate_puzzle(77), generate_puzzle(77))
    assert not _puzzles_equal(generate_puzzle(77), generate_puzzle(78))
    puzzle = generate_puzzle(905, 2, 1)
    training = [{k:v.copy() for k,v in p.items()} for p in puzzle["train"]]
    prediction = solve(training, puzzle["test"][0]["input"].copy())
    assert np.array_equal(prediction,puzzle["test"][0]["output"])
    damaged = {s:[{k:v.copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect answer accepted")
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(9,a,b)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    _diagnostic_test()


def _diagnostic_test():
    # Same small cue, different demonstrated outer boundaries: the reference
    # must change the answer. No target-only diagnostic can solve both.
    rr, cc = np.indices((6,6))
    d = abs(rr-2)+abs(cc-2)
    cue = np.where(d < 2, 1+d%2, 0).astype(np.int8)
    outputs = []
    for radius in (2,3,4):
        outer = np.where(d >= radius, 1+d%2, 0).astype(np.int8)
        choices = _prediction_candidates([cue,outer], cue, 0)
        assert len(choices) == 1
        answer = next(iter(choices.values()))[0]
        assert np.array_equal(answer, outer)
        assert np.array_equal(_solve_scene(dict(reference=[outer,cue], occupied=cue, neutral=0)), answer)
        outputs.append(_key(answer))
    assert len(set(outputs)) == 3
    # The same small corner cue can designate different symmetric blueprints.
    cue = np.zeros((6,6), dtype=np.int8)
    cue[0,1], cue[0,2], cue[1,0] = 2,1,2
    full = np.ones((6,6), dtype=np.int8)
    full[0,0] = full[0,-1] = full[-1,0] = full[-1,-1] = 0
    for r,c in ((0,1),(0,4),(1,0),(1,5),(4,0),(4,5),(5,1),(5,4)):
        full[r,c] = 2
    other = full.copy()
    other[2:4,0] = other[2:4,-1] = 0
    answers = [_solve_scene(dict(reference=[cue,b], occupied=cue, neutral=0)) for b in (full,other)]
    assert not np.array_equal(*answers)
    # A complete pair must choose which aligned cue sets to connect.
    cue = np.zeros((6,6),dtype=np.int8)
    cue[0,0], cue[0,5], cue[4,0] = 2,1,1
    answers = []
    for axes in ((0,), (1,), (0,1)):
        full = _fill_cues(cue,0,1,axes)
        answers.append(_solve_scene(dict(reference=[cue,full], occupied=cue, neutral=0)))
    assert len({_key(x) for x in answers}) == 3
    # Preserve all original cues, fill intersections with primary, transpose.
    cue = np.zeros((6,6), dtype=np.int8)
    cue[0,0], cue[0,4], cue[4,0], cue[5,5] = 2,1,1,1
    full = _fill_cues(cue, 0, 1, (0,1))
    assert full[0,2] == full[2,0] == 1 and full[5,5] == 1 and full[0,0] == 2
    assert np.array_equal(_fill_cues(cue.T,0,1,(0,1)), full.T)
    # An unobservable remapping of equally rare endpoint roles is rejected.
    a = np.array([[1,0,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=np.int8)
    b = _fill_cues(a,0,1,(0,))
    target = np.where(a == 1,3,np.where(a == 2,4,0)).astype(np.int8)
    # Exact geometry DOES resolve these endpoints; a different target geometry
    # with no shared role must not be accepted by frequency alone.
    target = target.T.copy()
    assert len(_prediction_candidates([a,b],target,0)) != 1
    # Directly exercise the bounded constructions, too.
    for i,kind in enumerate(("line","complement","distance","symmetry")):
        pair = _fallback_pair(np.random.default_rng(i),kind,dict(colors=np.arange(10)))
        assert np.array_equal(_solve_scene(_extract_scene(pair["input"])),pair["output"])


def _distance_features(reference, neutral):
    h, w = reference[0].shape
    rr, cc = np.indices((h,w))
    centers = [((r,r),(c,c)) for r in range(h) for c in range(w)]
    centers.insert(0, (((h-1)//2,h//2), ((w-1)//2,w//2)))
    for rows, cols in centers:
        d = np.minimum(abs(rr-rows[0]),abs(rr-rows[1])) + np.minimum(abs(cc-cols[0]),abs(cc-cols[1]))
        for a,b in (reference, reference[::-1]):
            extent = int(d[a != neutral].max())+1
            if not np.array_equal(a != neutral, d < extent) or not np.any(b != neutral):
                continue
            radius = int(d[b != neutral].min())
            if radius < extent or not np.array_equal(b != neutral, d >= radius):
                continue
            combined = np.where(a != neutral,a,b)
            known = combined != neutral
            roles = [np.unique(combined[known & (d%2 == p)]) for p in (0,1)]
            if any(len(x) != 1 for x in roles) or roles[0][0] == roles[1][0]:
                continue
            if any(len(set(map(int,np.unique(x))) - {neutral}) != 2 for x in (a,b)):
                continue
            return dict(radius=radius, seed_extent=extent, center=list(rows+cols))
    return None


def review_features(pair):
    """Task-specific measurements reconstructed from returned pixels only."""
    grid, out = pair["input"], pair["output"]
    scene = _extract_scene(grid)
    neutral = scene["neutral"]
    ref, target = scene["reference"], scene["occupied"]
    ref_colors = set(map(int,np.unique(ref[0]))) - {neutral}
    target_colors = set(map(int,np.unique(target))) - {neutral}
    distance = _distance_features(ref, neutral)
    sparse, full = sorted(ref, key=lambda x: np.count_nonzero(x != neutral))
    complement = len(ref_colors) == 1 and np.all((ref[0] != neutral) != (ref[1] != neutral))
    symmetric = (np.array_equal(full,full[::-1]) and np.array_equal(full,full[:,::-1])
                 and np.all((sparse == neutral) | (sparse == full))
                 and np.count_nonzero(full != neutral) > np.count_nonzero(sparse != neutral))
    kind = "complement" if complement else "distance" if distance else "symmetry" if symmetric else "line"
    orientations, gaps, offsets, padding = [], [], [], []
    group_colors, frame_colors = [], []
    for group, box in zip(scene["panels"], scene["boxes"]):
        a,b = group
        horizontal = a[1]+a[3] <= b[1] or b[1]+b[3] <= a[1]
        axis, cross = (1,0) if horizontal else (0,1)
        a,b = sorted(group, key=lambda x:x[axis])
        orientations.append("h" if horizontal else "v")
        gaps.append(b[axis]-a[axis]-a[axis+2])
        offsets.append(abs(b[cross]-a[cross]))
        group_colors.append(box[4])
        frame_colors.append(a[4])
        for panel in group:
            padding.extend([panel[0]-box[0],panel[1]-box[1],
                            box[0]+box[2]-panel[0]-panel[2], box[1]+box[3]-panel[1]-panel[3]])
    outside = np.ones(grid.shape, bool)
    for r,c,h,w,_ in scene["boxes"]:
        outside[r:r+h,c:c+w] = False
    exterior_palette = set(map(int,np.unique(grid[outside])))
    equal, adjacent = 0, 0
    for x,y,allowed in ((grid[:,:-1],grid[:,1:],outside[:,:-1]&outside[:,1:]),
                         (grid[:-1],grid[1:],outside[:-1]&outside[1:])):
        equal += int(np.count_nonzero((x == y)&allowed))
        adjacent += int(np.count_nonzero(allowed))
    normalized = {neutral:0}
    for color in out.flat:
        if int(color) not in normalized:
            normalized[int(color)] = len(normalized)
    result = dict(kind=kind, shape=list(scene["shape"]), canvas=list(grid.shape), neutral=neutral,
                  total_colors=len(np.unique(grid)), exterior_colors=len(exterior_palette),
                  exterior_repeats=equal, exterior_repeat_rate=round(equal/max(1,adjacent),4),
                  exterior_role_overlap=len(exterior_palette & (ref_colors|target_colors|{neutral})),
                  exterior_irrelevant_colors=len(exterior_palette - ref_colors-target_colors-{neutral}-set(frame_colors)-set(group_colors)),
                  neutral_in_exterior=neutral in exterior_palette, orientations=orientations,
                  mixed=orientations[0] != orientations[1], gaps=gaps, offsets=offsets,
                  padding=padding, boxes=[list(b[:4]) for b in scene["boxes"]],
                  unequal_boxes=scene["boxes"][0][2:4] != scene["boxes"][1][2:4],
                  edge_contact=any(r==0 or c==0 or r+h==grid.shape[0] or c+w==grid.shape[1]
                                   for r,c,h,w,_ in scene["boxes"]),
                  spanning_frame=any(h==grid.shape[0] or w==grid.shape[1] for _,_,h,w,_ in scene["boxes"]),
                  group_colors=group_colors, frame_colors=frame_colors,
                  different_group_colors=group_colors[0] != group_colors[1],
                  different_panel_colors=frame_colors[0] != frame_colors[1],
                  frame_motif_reuse=bool(set(frame_colors) & (ref_colors|target_colors)),
                  group_motif_reuse=bool(set(group_colors) & (ref_colors|target_colors)),
                  palette_mode="same" if ref_colors==target_colors else "shared" if ref_colors&target_colors else "remapped",
                  reference_group=scene["reference_group"], occupied_slot=scene["occupied_slot"],
                  output_density=round(float(np.mean(out != neutral)),4),
                  output_mask=(out != neutral).astype(np.uint8).tobytes().hex(),
                  normalized_output=_recolor(out,normalized).tobytes().hex())
    if kind == "distance":
        result.update(distance)
        r0,r1,c0,c1 = distance["center"]
        rr,cc = np.indices(target.shape)
        d = np.minimum(abs(rr-r0),abs(rr-r1))+np.minimum(abs(cc-c0),abs(cc-c1))
        result["distance_direction"] = "inner_to_outer" if np.array_equal(target != neutral,d < distance["seed_extent"]) else "outer_to_inner"
    if kind == "complement":
        mask = target != neutral
        result["occupied_density"] = round(float(mask.mean()),4)
        todo = set(map(tuple,np.argwhere(mask)))
        components = 0
        while todo:
            stack = [todo.pop()]
            components += 1
            while stack:
                r,c = stack.pop()
                for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if p in todo:
                        todo.remove(p)
                        stack.append(p)
        result["occupied_components"] = components
    if kind in ("line", "symmetry"):
        occupancy = sparse != neutral
        segments, lengths, segment_axes = [], [], []
        touched = np.zeros(occupancy.shape,bool)
        row_fill, col_fill = np.zeros_like(touched), np.zeros_like(touched)
        for axis in (0,1):
            for row, values in enumerate(occupancy if axis == 0 else occupancy.T):
                cols = np.flatnonzero(values)
                if len(cols) >= 2 and cols[-1]-cols[0] >= 2:
                    lo, hi = int(cols[0]), int(cols[-1])
                    sl = (row,slice(lo,hi+1)) if axis == 0 else (slice(lo,hi+1),row)
                    touched[sl] = True
                    (row_fill if axis == 0 else col_fill)[sl] = True
                    segments.append((axis,row,lo,hi))
                    segment_axes.append(axis)
                    lengths.append(hi-lo+1)
        result.update(cue_count=int(occupancy.sum()), aligned_sets=len(segments),
                      connected_sets=sum(bool(np.all(full[row,lo:hi+1] != neutral)) if axis == 0
                                         else bool(np.all(full[lo:hi+1,row] != neutral))
                                         for axis,row,lo,hi in segments),
                      segment_lengths=lengths, line_axes=sorted(set(segment_axes)),
                      isolated_cues=int(np.count_nonzero(occupancy & ~touched)),
                      intersections=int(np.count_nonzero(row_fill & col_fill & ~occupancy)),
                      bends=int(np.count_nonzero(row_fill & col_fill & occupancy)),
                      missing_rows=int(np.count_nonzero(~occupancy.any(axis=1))),
                      missing_columns=int(np.count_nonzero(~occupancy.any(axis=0))),
                      missing_mirrors=not (np.array_equal(sparse,sparse[::-1]) and np.array_equal(sparse,sparse[:,::-1])),
                      occupied_filled=int(np.count_nonzero(target != neutral)) > int(np.count_nonzero(out != neutral)))
        if kind == "symmetry":
            result["endpoint_axes"] = [axis for axis in (0,1) if np.array_equal(_endpoints(full,neutral,axis),sparse)]
        remaining = set(map(tuple,np.argwhere(full != neutral)))
        changed_components = 0
        while remaining:
            stack = [remaining.pop()]
            changed = False
            while stack:
                r,c = stack.pop()
                changed |= bool(sparse[r,c] == neutral)
                for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
            changed_components += changed
        result['connection_components'] = changed_components
    return result


def _review_self_test(records):
    rows = [f for _,_,f in records]
    by_kind = {k:[f for f in rows if f['kind'] == k] for k in ('line','complement','distance','symmetry')}
    for split in ('train','test'):
        assert {f['kind'] for _,s,f in records if s == split} == set(by_kind)
    for key in ('mixed','unequal_boxes','edge_contact','spanning_frame','different_group_colors',
                'different_panel_colors','frame_motif_reuse','group_motif_reuse','neutral_in_exterior'):
        assert {f[key] for f in rows} == {False,True}, key
    assert {f['palette_mode'] for f in rows} == {'same','shared','remapped'}
    assert max(x for f in rows for x in f['gaps']) >= 9
    assert max(x for f in rows for x in f['offsets']) >= 3
    assert max(x for f in rows for x in f['padding']) >= 5
    assert len({tuple(f['group_colors']) for f in rows}) >= 20
    assert len({tuple(f['frame_colors']) for f in rows}) >= 20
    assert len({f['exterior_colors'] for f in rows}) >= 5
    assert any(f['exterior_irrelevant_colors'] for f in rows)
    assert any(f['exterior_repeats'] >= 10 for f in rows)
    assert {f['reference_group'] for f in rows} == {0,1}
    assert {f['occupied_slot'] for f in rows} == {0,1}
    line = by_kind['line']
    assert {tuple(f['line_axes']) for f in line} >= {(0,), (1,), (0,1)}
    assert max(f['cue_count'] for f in line) >= 6
    assert max(f['connected_sets'] for f in line) >= 3
    assert any(f['bends'] for f in line) and any(f['intersections'] for f in line)
    assert any(f['isolated_cues'] for f in line)
    assert any(len(set(f['segment_lengths'])) > 1 for f in line)
    assert any(f['occupied_filled'] for f in line)
    assert any(f['connection_components'] >= 2 for f in line)
    assert any(f['shape'][0] != f['shape'][1] for f in line)
    symmetry = by_kind['symmetry']
    assert any(f['missing_rows'] >= 3 and f['missing_mirrors'] for f in symmetry)
    assert any(0 in f['endpoint_axes'] for f in symmetry) and any(1 in f['endpoint_axes'] for f in symmetry)
    assert len({(tuple(f['shape']),f['output_mask']) for f in symmetry}) >= 30
    distance = by_kind['distance']
    assert len({f['radius'] for f in distance}) >= 4
    assert len({f['seed_extent'] for f in distance}) >= 2
    assert len({tuple(f['center']) for f in distance}) >= 12
    assert any(f['radius'] > f['seed_extent'] for f in distance)
    assert len({(tuple(f['shape']),f['output_mask']) for f in distance}) >= 30
    assert len({(tuple(f['shape']),f['normalized_output']) for f in distance}) >= 40
    assert {f['distance_direction'] for f in distance if f['palette_mode'] == 'remapped'} == {'inner_to_outer','outer_to_inner'}
    complement = by_kind['complement']
    assert min(f['occupied_density'] for f in complement) < .15
    assert max(f['occupied_density'] for f in complement) > .85
    assert any(f['occupied_components'] == 1 for f in complement)
    assert any(f['shape'][0] != f['shape'][1] for f in complement)
    # Both stable and varying puzzles, without imposing a new semantic rule.
    for key in ('neutral','orientations','palette_mode','shape','gaps','frame_colors'):
        diversity = [len({str(f[key]) for s,_,f in records if s == seed}) for seed in range(42)]
        assert max(diversity) >= 3, key
        if key in ('neutral','orientations'):
            assert min(diversity) == 1, key
    regimes = set()
    for seed in range(42):
        train = {f['kind'] for s,split,f in records if s == seed and split == 'train'}
        test = {f['kind'] for s,split,f in records if s == seed and split == 'test'}
        regimes.add((tuple(sorted(train)),tuple(sorted(test))))
    assert len(regimes) >= 12
