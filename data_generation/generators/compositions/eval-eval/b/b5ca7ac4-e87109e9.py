"""Optical dispatch: turn beams at cargo, then pack cargo in the outgoing headings.

The top key is local to each input. Rays use a snapshot of the input geometry;
the output retains that light layer and stamps docked cargo over it. Only the
key is removed; light never changes a carried payload.
No parent code, files, examples, or generator state are used by solve.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_D = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N E S W
_HOST_SEEDS = (0, 667611400, 860015884, 325483261, 123769699, 760939904,
    945313681, 763868249, 727735378, 642724387, 727068123, 369429534,
    190574211, 5547911, 678829243, 760129163, 639915220, 632015295,
    929950071, 790588937, 874801232, 571391719, 640562817, 87369243,
    572895512, 863968068, 48826456, 980509406, 711405264, 302210685,
    295643011, 713104004, 649455502, 439777856, 495713993, 961797443,
    331270679, 455184107, 904514996, 51504290, 183327202, 962000445)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "invalid grid shape")
    _need(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)),
          "grid must contain integer ARC colors")
    return g.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for name, x, lo, hi in (("num_train", num_train, 2, 10),
                            ("num_test", num_test, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lo <= x <= hi:
            raise ValueError(name + " is outside its legal range")
    _need(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _overlap(a, b):
    return (a['r'] < b['r'] + b['h'] and b['r'] < a['r'] + a['h']
            and a['c'] < b['c'] + b['w'] and b['c'] < a['c'] + a['w'])


def _decode(input_grid):
    """Find the visible key, complete frames, and remaining solid source."""
    g = _grid(input_grid)
    border = int(g[0, 0])
    _need(np.all(g[0] == border), "missing key top border")
    rows = np.flatnonzero(np.all(g[1:] == border, axis=1)) + 1
    _need(len(rows) > 0, "missing key bottom border")
    bottom = int(rows[0])
    _need(2 <= bottom < len(g) - 1, "invalid key height")
    band = g[1:bottom]
    _need(np.all(band == band[0]), "key bars must span the interior height")
    row = band[0]
    _need(row[0] == row[-1] == border, "key needs side walls")
    panels = []
    c = 0
    while c < len(row):
        if row[c] == border:
            c += 1
            continue
        a = c
        while c < len(row) and row[c] != border:
            c += 1
        panels.append((a, c))
    _need(bool(panels) and all(b - a >= 3 for a, b in panels), "ambiguous key panels")
    background = int(row[panels[0][0] + 1])
    _need(background != border, "key interior must differ from border")
    turns = {}
    for a, b in panels:
        _need(np.all(row[a + 1:b - 1] == background), "inconsistent key interior")
        l, r = int(row[a]), int(row[b - 1])
        _need((l == background) != (r == background), "one side bar per panel required")
        color, turn = (r, 1) if l == background else (l, -1)
        _need(color not in turns and color not in (background, border), "conflicting key color")
        turns[color] = turn
    field = g[bottom + 1:].copy()
    h, w = field.shape
    covered = field == background
    objects = []
    # Whole-footprint marking permits adjacent frames and keys reused in payloads.
    for r in range(h):
        for c in range(w):
            if covered[r, c] or int(field[r, c]) not in turns:
                continue
            f = int(field[r, c])
            _need(r + 2 < h and c + 2 < w, "incomplete cargo frame")
            _need(field[r + 1, c + 1] not in (f, background), "erased cargo interior")
            rr, cc = r + 2, c + 2
            while rr < h and field[rr, c + 1] != f:
                rr += 1
            while cc < w and field[r + 1, cc] != f:
                cc += 1
            _need(rr < h and cc < w, "open cargo frame")
            patch = field[r:rr + 1, c:cc + 1]
            _need(np.all(patch[[0, -1], :] == f) and np.all(patch[:, [0, -1]] == f),
                  "cargo border must be complete")
            inside = patch[1:-1, 1:-1]
            _need(np.all(inside != f) and np.all(inside != background), "ambiguous cargo interior")
            _need(not np.any(covered[r:rr + 1, c:cc + 1]), "overlapping cargo")
            covered[r:rr + 1, c:cc + 1] = True
            objects.append(dict(r=r, c=c, h=rr - r + 1, w=cc - c + 1,
                                f=f, patch=patch.copy()))
    _need(bool(objects), "at least one cargo object required")
    cells = np.argwhere(~covered)
    _need(len(cells) > 0, "missing source")
    r, c = map(int, cells.min(axis=0))
    rr, cc = map(int, cells.max(axis=0) + 1)
    k = rr - r
    source_color = int(field[r, c])
    _need(k == cc - c and len(cells) == k * k
          and np.all(field[r:rr, c:cc] == source_color), "source must be one solid square")
    _need(source_color not in turns and source_color not in (background, border),
          "source color must be unkeyed and distinct")
    _need(all(not np.any(o['patch'] == source_color) for o in objects),
          "source color cannot occur in cargo")
    _need(np.count_nonzero(field == background) * 2 > field.size,
          "field background must occupy over half the input field")
    return dict(field=field, shape=field.shape, background=background, border=border,
                source=(r, c, k), source_color=source_color, objects=objects,
                turns=turns, crop=bottom + 1, panel_widths=tuple(b - a + 2 for a, b in panels))


def _trace(shape, source, objects, turns):
    """Trace four square cross sections against original cargo rectangles.

    Return the assigned headings, all collision events, and straight segments.
    Repeated encounters are legal only when their outgoing headings agree.
    """
    h, w = shape
    sr, sc, k = source
    assigned = [None] * len(objects)
    events, segments = [], []
    for branch in range(4):
        r, c, d = sr, sc, branch
        seen = set()
        previous_hit = False
        while True:
            _need((r, c, d) not in seen, "cyclic ray route")
            seen.add((r, c, d))
            limit = (r, w - c - k, h - r - k, c)[d]
            hits = []
            for i, o in enumerate(objects):
                a, b, e, f = o['r'], o['r'] + o['h'], o['c'], o['c'] + o['w']
                if d % 2 == 0:
                    if e >= c + k or f <= c:
                        continue
                    ahead = b <= r if d == 0 else a >= r + k
                    distance = r - b if d == 0 else a - r - k
                    full = e <= c and f >= c + k
                else:
                    if a >= r + k or b <= r:
                        continue
                    ahead = e >= c + k if d == 1 else f <= c
                    distance = e - c - k if d == 1 else c - f
                    full = a <= r and b >= r + k
                if ahead and distance < limit:
                    hits.append((distance, i, full))
            if hits:
                length = min(x[0] for x in hits)
                nearest = [x for x in hits if x[0] == length]
                _need(len(nearest) == 1 and nearest[0][2], "partial or simultaneous cargo impact")
                _need(not previous_hit or length > 0, "no clear travel between turns")
                hit = nearest[0][1]
            else:
                length, hit = limit, -1
            dr, dc = _D[d]
            nr, nc = r + dr * length, c + dc * length
            swept = dict(r=min(r, nr), c=min(c, nc),
                         h=abs(nr - r) + k, w=abs(nc - c) + k)
            _need(not any(_overlap(swept, o) for o in objects), "blocked exterior turning square")
            segments.append((branch, r, c, d, length, hit))
            if hit < 0:
                break
            nd = (d + turns[objects[hit]['f']]) % 4
            _need(assigned[hit] in (None, nd), "conflicting dispatch directions")
            assigned[hit] = nd
            events.append((hit, branch, d, nd, nr, nc))
            r, c, d = nr, nc, nd
            previous_hit = True
    _need(all(d is not None for d in assigned), "every cargo must be reached")
    return assigned, events, segments


def _pack(shape, objects, directions, edge_first=False):
    """Stable compaction; distinct heading classes pass through each other."""
    result = [dict(o, d=int(d)) for o, d in zip(objects, directions)]
    for d in range(4):
        ax, size, perp, ps = (('r', 'h', 'c', 'w') if d % 2 == 0 else ('c', 'w', 'r', 'h'))
        length = shape[d % 2]
        low = d in (0, 3)
        group = [o for o in result if o['d'] == d]
        group.sort(key=lambda o: (o[ax] if low else -o[ax] - o[size], o[perp]))
        settled = []
        for o in group:
            blockers = [p for p in settled if o[perp] < p[perp] + p[ps]
                        and p[perp] < o[perp] + o[ps]]
            if edge_first:  # A competing rule used only for demonstration checks.
                coords = range(length - o[size] + 1)
                if not low:
                    coords = reversed(coords)
                for coord in coords:
                    o[ax] = coord
                    if not any(_overlap(o, p) for p in settled):
                        break
                else:
                    raise ValueError("packing capacity exceeded")
            elif low:
                o[ax] = max((p[ax] + p[size] for p in blockers), default=0)
            else:
                o[ax] = min((p[ax] for p in blockers), default=length) - o[size]
            _need(0 <= o[ax] <= length - o[size], "packing capacity exceeded")
            settled.append(o)
    _need(not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]),
          "packed heading classes overlap")
    return result


def _render(shape, background, objects, base=None):
    out = np.full(shape, background, dtype=np.int8) if base is None else base.copy()
    for o in objects:
        out[o['r']:o['r'] + o['h'], o['c']:o['c'] + o['w']] = o['patch']
    return out


def _light_layer(q, segments):
    """Snapshot light, including its beacon, on an otherwise empty field.

    Tracing has checked that each swept square avoids all original cargo.
    The later cargo layer may cover light, including the original beacon.
    """
    light = np.full(q['shape'], q['background'], dtype=np.int8)
    k = q['source'][2]
    for _, r, c, d, length, _ in segments:
        dr, dc = _D[d]
        nr, nc = r + dr * length, c + dc * length
        light[min(r, nr):max(r, nr) + k, min(c, nc):max(c, nc) + k] = q['source_color']
    return light


def solve(input_grid: np.ndarray) -> np.ndarray:
    q = _decode(input_grid)
    directions, _, segments = _trace(q['shape'], q['source'], q['objects'], q['turns'])
    return _render(q['shape'], q['background'], _pack(q['shape'], q['objects'], directions),
                   _light_layer(q, segments))


def _features(q):
    objects = q['objects']
    directions, events, segments = _trace(q['shape'], q['source'], objects, q['turns'])
    settled = _pack(q['shape'], objects, directions)
    try:
        old = _pack(q['shape'], objects, directions, edge_first=True)
        witness = [(o['r'], o['c']) for o in old] != [(o['r'], o['c']) for o in settled]
    except ValueError:
        witness = True
    edges, sharing = 0, False
    for i, a in enumerate(objects):
        for j, b in enumerate(objects[:i]):
            if directions[i] != directions[j]:
                continue
            perp, ps = ('c', 'w') if directions[i] % 2 == 0 else ('r', 'h')
            collides = a[perp] < b[perp] + b[ps] and b[perp] < a[perp] + a[ps]
            edges += int(collides)
            ax = 'r' if directions[i] % 2 == 0 else 'c'
            sharing |= not collides and settled[i][ax] == settled[j][ax]
    moved = [(o['f'], d) for o, p, d in zip(objects, settled, directions)
             if (o['r'], o['c']) != (p['r'], p['c'])]
    return dict(directions=directions, events=events, segments=segments, settled=settled,
                witness=witness, edges=edges, sharing=sharing, moved=moved,
                max_turns=max(Counter(e[1] for e in events).values(), default=0))


def _grow(rng, k, turns, target, shape=None, source=None, objects=None,
          keep_witness=False):
    if shape is None:
        shape = (int(rng.integers(max(14, 4 * k + 3), 27)), int(rng.integers(20, 31)))
    if source is None:
        source = (int(rng.integers(0, shape[0] - k + 1)),
                  int(rng.integers(0, shape[1] - k + 1)), k)
    objects = [] if objects is None else [dict(o) for o in objects]
    source_box = dict(r=source[0], c=source[1], h=k, w=k)
    _, _, segments = _trace(shape, source, objects, turns)
    # Failed insertions do not alter the scene. The target is a ceiling, not a quota.
    for _ in range(max(1, target - len(objects)) * 50):
        if len(objects) >= target:
            break
        choices = [s for s in segments if s[4] >= 2]
        if not choices:
            break
        recursive = [s for s in choices if s[3] != s[0]]
        if recursive and rng.random() < .6:
            choices = recursive
        branch, r, c, d, length, _ = choices[int(rng.integers(len(choices)))]
        distance = int(rng.integers(1, length))
        dr, dc = _D[d]
        r, c = r + dr * distance, c + dc * distance
        hh, ww = int(rng.integers(max(3, k), 8)), int(rng.integers(max(3, k), 8))
        if d % 2 == 0:
            a = r - hh if d == 0 else r + k
            b = c - int(rng.integers(ww - k + 1))
        else:
            a = r - int(rng.integers(hh - k + 1))
            b = c + k if d == 1 else c - ww
        obj = dict(r=a, c=b, h=hh, w=ww, f=int(rng.choice(list(turns))))
        if not (0 <= a <= shape[0] - hh and 0 <= b <= shape[1] - ww):
            continue
        if any(_overlap(obj, p) for p in objects + [source_box]):
            continue
        candidate = objects + [obj]
        if 2 * (sum(p['h'] * p['w'] for p in candidate) + k * k) >= shape[0] * shape[1]:
            continue
        try:
            directions, _, new_segments = _trace(shape, source, candidate, turns)
            settled = _pack(shape, candidate, directions)
            if keep_witness:
                old = _pack(shape, candidate, directions, edge_first=True)
                if [(p['r'], p['c']) for p in old] == [(p['r'], p['c']) for p in settled]:
                    continue
        except ValueError:
            continue
        objects, segments = candidate, new_segments
    return shape, source, objects


def _rotate_scene(shape, source, objects, times):
    # Clockwise quarter rotations keep relative key meanings unchanged.
    for _ in range(times):
        h, w = shape
        sr, sc, k = source
        source = (sc, h - sr - k, k)
        objects = [dict(o, r=o['c'], c=h - o['r'] - o['h'], h=o['w'], w=o['h'])
                   for o in objects]
        shape = (w, h)
    return shape, source, objects


def _lesson(rng, k, turns):
    """Parameterized recursive overlap-chain witness, in any orientation.

    Three north-dispatched boxes form a chain whose first and last column
    spans are disjoint. Relay boxes create the requisite incoming headings.
    The west branch and original north branch merge, teaching consistent re-hit.
    """
    left = next(c for c in turns if turns[c] == -1)
    right = next(c for c in turns if turns[c] == 1)
    cargo_width = int(rng.integers(3, 5))
    step = cargo_width - 1
    base = int(rng.integers(11, 15))
    third = base + 2 * step
    min_width = third + cargo_width + 3
    if min_width > 27:
        base -= min_width - 27
        third = base + 2 * step
        min_width = 27
    top = int(rng.integers(0, 2))
    top_h = 3
    bottom_top = int(rng.integers(max(4, top + top_h + k + 1), 8))
    bottom_h = int(rng.integers(3, 5))
    middle_top = bottom_top + bottom_h - 1
    cargo_h = k + 2
    sr = middle_top + cargo_h + int(rng.integers(2, 5))
    min_height = max(20, sr + k + 1)
    shape = (int(rng.integers(min_height, 28)), int(rng.integers(min_width, 28)))
    source = (sr, 6, k)
    def o(r, c, h, w, f):
        return dict(r=r, c=c, h=h, w=w, f=f)
    objects = [o(sr - 1, base, cargo_h, cargo_width, left),
               o(bottom_top, base - k - 1, bottom_h, 3, right),
               o(middle_top, base + step, cargo_h, cargo_width, left),
               o(top, base + step - k - 1, top_h, 3, right),
               o(top + top_h - 1, third, cargo_h, cargo_width, left),
               o(sr - 1, 1, cargo_h, 3, right),
               o(bottom_top, 3, bottom_h, 5, left)]
    directions, _, _ = _trace(shape, source, objects, turns)
    _pack(shape, objects, directions)
    return _rotate_scene(shape, source, objects, int(rng.integers(4)))


def _make_pair(rng, scene, turns, frame_colors, patterned=False):
    shape, source, objects = scene
    _need(bool(objects), "empty generated scene")
    ordinary = [c for c in range(10) if c not in frame_colors]
    palette = list(map(int, rng.permutation(ordinary)))
    background, source_color, border = palette[:3]
    payloads = palette[3:]
    directions, _, segments = _trace(shape, source, objects, turns)
    # A shared payload across different headings explicitly refutes payload sorting.
    shared = payloads[0]
    for o in objects:
        interior = np.full((o['h'] - 2, o['w'] - 2), int(rng.choice(payloads)), dtype=np.int8)
        if patterned and interior.size > 1 and rng.random() < .6:
            rr, cc = np.indices(interior.shape)
            colors = rng.choice(payloads, 2, replace=False)
            interior = colors[(rr + cc) % 2].astype(np.int8)
        o['patch'] = np.full((o['h'], o['w']), o['f'], dtype=np.int8)
        o['patch'][1:-1, 1:-1] = interior
    first = 0
    other = next((i for i, d in enumerate(directions) if d != directions[first]), None)
    if other is not None:
        for i in (first, other):
            objects[i]['patch'][1:-1, 1:-1] = shared
    field = _render(shape, background, objects)
    sr, sc, k = source
    field[sr:sr + k, sc:sc + k] = source_color
    count = len(turns)
    _need(shape[1] >= 5 * count, "key panels do not fit")
    widths = np.full(count, 5, dtype=int)
    widths += rng.multinomial(shape[1] - 5 * count, np.full(count, 1 / count))
    header_height = int(rng.integers(3, min(5, 30 - shape[0]) + 1))
    header = np.full((header_height, shape[1]), background, dtype=np.int8)
    header[[0, -1], :] = border
    start = 0
    for color, width in zip(rng.permutation(list(turns)), widths):
        width = int(width)
        header[:, start] = header[:, start + width - 1] = border
        bar = start + 1 if turns[int(color)] == -1 else start + width - 2
        header[1:-1, bar] = color
        start += width
    light_info = dict(shape=shape, source=source, background=background, source_color=source_color)
    output = _render(shape, background, _pack(shape, objects, directions),
                     _light_layer(light_info, segments))
    return dict(input=np.vstack((header, field)), output=output)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
          and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    keys = list(map(int, rng.choice(10, int(rng.integers(2, 5)), replace=False)))
    # Keys are stable identities within a puzzle; each input encodes fresh meanings.
    lesson_turns = {keys[0]: -1, keys[1]: 1}
    lesson_turns.update({c: int(rng.choice((-1, 1))) for c in keys[2:]})
    if rng.random() < .5:
        lesson_turns = {c: -v for c, v in lesson_turns.items()}
    lesson_k = int(rng.integers(1, 3))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(60):
                if split == 'train' and index == 0:
                    turns = lesson_turns.copy()
                    scene = _lesson(rng, lesson_k, turns)
                    if rng.random() < .5:
                        scene = _grow(rng, lesson_k, turns, int(rng.integers(8, 11)),
                                      *scene, keep_witness=True)
                else:
                    k = 3 if split == 'test' else 3 - lesson_k if index == 1 else int(rng.integers(1, 3))
                    turns = ({c: -v for c, v in lesson_turns.items()} if index == 1 and split == 'train'
                             else {c: int(rng.choice((-1, 1))) for c in keys})
                    target = int(rng.integers(2, 9)) if split == 'train' else int(rng.integers(3, 11))
                    scene = _grow(rng, k, turns, target)
                    if not scene[2]:
                        continue
                pair = _make_pair(rng, scene, turns, keys, patterned=index > 0 or rng.random() < .5)
                q = _decode(pair['input'])
                info = _features(q)
                # All tests may exercise the stable chain convention because it was
                # shown in training. No feature needs an index-dependent new rule.
                if split == 'train' and index == 0 and not info['witness']:
                    continue
                sig = _signature(pair['input'])
                if sig not in seen:
                    puzzle[split].append(pair)
                    seen.add(sig)
                    break
            else:
                raise RuntimeError("bounded generation exhausted unique valid scenes")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_trace(q):
    """Supplemental unit-step check of ray geometry, with no painted obstacles."""
    occ = np.full(q['shape'], -1, dtype=int)
    for i, o in enumerate(q['objects']):
        occ[o['r']:o['r'] + o['h'], o['c']:o['c'] + o['w']] = i
    directions = [None] * len(q['objects'])
    sr, sc, k = q['source']
    h, w = q['shape']
    for branch in range(4):
        r, c, d = sr, sc, branch
        seen = set()
        while (r, w - c - k, h - r - k, c)[d] > 0:
            _need((r, c, d) not in seen, "cell tracer cycle")
            seen.add((r, c, d))
            front = (occ[r - 1, c:c + k] if d == 0 else occ[r:r + k, c + k] if d == 1
                     else occ[r + k, c:c + k] if d == 2 else occ[r:r + k, c - 1])
            if np.any(front >= 0):
                _need(np.all(front == front[0]) and front[0] >= 0, "cell tracer partial hit")
                hit = int(front[0])
                d = (d + q['turns'][q['objects'][hit]['f']]) % 4
                _need(directions[hit] in (None, d), "cell tracer dispatch conflict")
                directions[hit] = d
            else:
                dr, dc = _D[d]
                r, c = r + dr, c + dc
    _need(all(d is not None for d in directions), "cell tracer unhit cargo")
    return directions


def _unit_pack(shape, objects, directions):
    """Supplemental physical check: same-heading cargo cannot jump past cargo."""
    result = [dict(o, d=d) for o, d in zip(objects, directions)]
    for d in range(4):
        group = [o for o in result if o['d'] == d]
        dr, dc = _D[d]
        for _ in range(30 * max(1, len(group)) + 1):
            moved = False
            for o in group:
                candidate = dict(o, r=o['r'] + dr, c=o['c'] + dc)
                if (0 <= candidate['r'] <= shape[0] - o['h']
                        and 0 <= candidate['c'] <= shape[1] - o['w']
                        and not any(p is not o and _overlap(candidate, p) for p in group)):
                    o['r'], o['c'] = candidate['r'], candidate['c']
                    moved = True
            if not moved:
                break
        else:
            raise ValueError("unit packing failed to converge")
    _need(not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]),
          "unit packing final collision")
    return result


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    training_k, test_k = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            _need(sig not in seen, "duplicate input")
            seen.add(sig)
            q = _decode(inp)
            f = _features(q)
            _need(out.shape == q['shape'], "incorrect field crop")
            _need(f['directions'] == _cell_trace(q), "ray checks disagree")
            physical = _unit_pack(q['shape'], q['objects'], f['directions'])
            _need([(o['r'], o['c']) for o in physical]
                  == [(o['r'], o['c']) for o in f['settled']], "packing checks disagree")
            light = _light_layer(q, f['segments'])
            expected = _render(q['shape'], q['background'], f['settled'], light)
            _need(np.array_equal(expected, out), "incorrect cargo translation or payload")
            occupied = np.zeros(q['shape'], dtype=bool)
            for o in f['settled']:
                occupied[o['r']:o['r'] + o['h'], o['c']:o['c'] + o['w']] = True
            f['covered_ink'] = int(np.count_nonzero((light == q['source_color']) & occupied))
            _need(np.array_equal(out == q['source_color'], (light == q['source_color']) & ~occupied),
                  "light must lie below the intact cargo layer")
            _need(np.count_nonzero(out != q['background'])
                  == sum(o['h'] * o['w'] for o in q['objects'])
                     + np.count_nonzero((light == q['source_color']) & ~occupied),
                  "cargo or visible light area not conserved")
            for o in q['objects']:
                _need(np.all(light[o['r']:o['r'] + o['h'], o['c']:o['c'] + o['w']] == q['background']),
                      "snapshot beams entered an original cargo footprint")
            for o, p, d in zip(q['objects'], f['settled'], f['directions']):
                _need(o['c'] == p['c'] if d % 2 == 0 else o['r'] == p['r'],
                      "perpendicular coordinate changed")
                dr, dc = _D[d]
                _need((p['r'] - o['r']) * dr + (p['c'] - o['c']) * dc >= 0,
                      "cargo moved against its assigned heading")
            (training_k if split == 'train' else test_k).add(q['source'][2])
            if split == 'train':
                training.append((q, f))
    _need(training_k == {1, 2} and test_k == {3}, "generated scale shift missing")
    _need(any(f['witness'] for _, f in training), "stable-order witness missing in training")
    _need(any(f['sharing'] and f['edges'] for _, f in training), "edge sharing / collision lessons missing")
    _need(any(f['covered_ink'] > 0 for _, f in training), "cargo-over-light precedence is undemonstrated")
    headings = {-1: set(), 1: set()}
    moved_by_frame = {}
    for q, f in training:
        for i, _, incoming, _, _, _ in f['events']:
            headings[q['turns'][q['objects'][i]['f']]].add(incoming)
        for frame, d in f['moved']:
            moved_by_frame.setdefault(frame, set()).add(d)
    _need(all(len(s) >= 2 for s in headings.values()), "both relative turns need multiple headings")
    _need(any(len(s) >= 2 for s in moved_by_frame.values()), "static frame-to-edge shortcut not excluded")
    _need(any(f['max_turns'] >= 3 for _, f in training), "recursive dispatch lesson missing")
    _need(any(not np.array_equal(_render(q['shape'], q['background'], q['objects'], _light_layer(q, f['segments'])),
                                 _render(q['shape'], q['background'], f['settled'], _light_layer(q, f['segments'])))
              for q, f in training), "omitting packing leaves all labels unchanged")
    # Thickness must affect visible labels, not just an invisible route constraint.
    thin_fits = True
    for q, f in training:
        if q['source'][2] == 1:
            continue
        sr, sc, _ = q['source']
        thin_q = dict(q, source=(sr, sc, 1))
        try:
            dirs, _, segs = _trace(q['shape'], thin_q['source'], q['objects'], q['turns'])
            thin = _render(q['shape'], q['background'], _pack(q['shape'], q['objects'], dirs),
                           _light_layer(thin_q, segs))
            actual = _render(q['shape'], q['background'], f['settled'], _light_layer(q, f['segments']))
            thin_fits &= np.array_equal(thin, actual)
        except ValueError:
            thin_fits = False
    _need(not thin_fits, "one-cell beam shortcut not distinguished in training")
    # Only one interpretation of the two side glyphs may fit the training outputs.
    fitting = []
    for l in (-1, 1):
        for r in (-1, 1):
            fits = True
            for pair in puzzle['train']:
                q = _decode(pair['input'])
                turns = {c: (l if t == -1 else r) for c, t in q['turns'].items()}
                try:
                    dirs, _, segs = _trace(q['shape'], q['source'], q['objects'], turns)
                    pred = _render(q['shape'], q['background'], _pack(q['shape'], q['objects'], dirs),
                                   _light_layer(q, segs))
                except ValueError:
                    fits = False
                    break
                if not np.array_equal(pred, pair['output']):
                    fits = False
                    break
            if fits:
                fitting.append((l, r))
    _need(fitting == [(-1, 1)], "relative key meaning is not uniquely demonstrated")


def self_test() -> dict:
    """Bounded 42-seed checks and deliberate counterexamples; no file writes."""
    shapes, footprints, counts, geometry, palettes, headings, depths = set(), set(), set(), set(), set(), set(), set()
    fingerprints, repeat_hits, patterned = set(), 0, 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = tuple((_signature(p['input']), _signature(p['output']))
                            for s in ('train', 'test') for p in puzzle[s])
        fingerprints.add(fingerprint)
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                q = _decode(pair['input'])
                f = _features(q)
                _need(np.array_equal(solve(pair['input']), pair['output']), "self-test solve mismatch")
                shapes.add(q['shape'])
                counts.add(len(q['objects']))
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                geometry.add((q['shape'], q['source'], tuple((o['r'], o['c'], o['h'], o['w']) for o in q['objects'])))
                footprints.update((o['h'], o['w']) for o in q['objects'])
                headings.update(f['directions'])
                depths.add(f['max_turns'])
                repeat_hits += len(f['events']) > len(q['objects'])
                patterned += any(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in q['objects'])
    _need(len(fingerprints) == 42 and len(geometry) >= 150, "insufficient geometric diversity")
    _need(len(shapes) >= 35 and len(footprints) >= 15 and len(counts) >= 6,
          "shape, footprint, or count variation missing")
    _need(headings == set(range(4)) and len(depths) >= 4, "route variation missing")
    _need(patterned > 30 and repeat_hits >= 42, "pattern / consistent repeated-hit coverage missing")
    p = generate_puzzle(0, 2, 1)
    p2 = generate_puzzle(0, 2, 1)
    _need(all(np.array_equal(a[n], b[n]) for s in p for a, b in zip(p[s], p2[s])
              for n in ('input', 'output')), "seed determinism failed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(731, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a wrong label and the specific greedy packing alternative.
    wrong = {s: [{n: np.asarray(g).copy() for n, g in pair.items()} for pair in p[s]] for s in p}
    witness_pair = next(pair for pair in wrong['train'] if _features(_decode(pair['input']))['witness'])
    q = _decode(witness_pair['input'])
    dirs, _, _ = _trace(q['shape'], q['source'], q['objects'], q['turns'])
    witness_pair['output'] = _render(q['shape'], q['background'], _pack(q['shape'], q['objects'], dirs, True),
                                     _light_layer(q, _features(q)['segments']))
    try:
        validate_generated_puzzle(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("greedy same-class overtaking accepted")
    # Reversing layer precedence must be caught even though the rays are correct.
    wrong = {s: [{n: g.copy() for n, g in pair.items()} for pair in p[s]] for s in p}
    witness_pair = next(pair for pair in wrong['train'] if _features(_decode(pair['input']))['witness'])
    q = _decode(witness_pair['input'])
    f = _features(q)
    light = _light_layer(q, f['segments'])
    witness_pair['output'][light == q['source_color']] = q['source_color']
    try:
        validate_generated_puzzle(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("light-over-cargo corruption accepted")
    # The solver is not restricted to the sampled 1/2/3 thickness schedule, and
    # keyed colors inside a different frame are payload, not additional relays.
    extra_rng = np.random.default_rng(1729)
    for _ in range(30):
        scene = _grow(extra_rng, 4, {2: -1, 3: 1}, 3)
        if scene[2]:
            break
    else:
        raise AssertionError("could not construct supplemental thickness-four scene")
    extra = _make_pair(extra_rng, scene, {2: -1, 3: 1}, [2, 3], patterned=True)
    q = _decode(extra['input'])
    q['objects'][0]['patch'][1:-1, 1:-1] = 3 if q['objects'][0]['f'] == 2 else 2
    sr, sc, k = q['source']
    field = _render(q['shape'], q['background'], q['objects'])
    field[sr:sr+k, sc:sc+k] = q['source_color']
    extra['input'][q['crop']:] = field
    dirs, _, segs = _trace(q['shape'], q['source'], q['objects'], q['turns'])
    expected = _render(q['shape'], q['background'], _pack(q['shape'], q['objects'], dirs),
                       _light_layer(q, segs))
    _need(np.array_equal(solve(extra['input']), expected), "broader input-only solving failed")
    # A partial thick-beam face, a cycle, and conflicting encounters are invalid.
    malformed = [((12, 12), (5, 5, 2), [dict(r=1, c=5, h=3, w=1, f=2)], {2: 1}),
                 ((10, 10), (4, 4, 1),
                  [dict(r=r, c=c, h=1, w=1, f=2) for r, c in ((4, 7), (7, 6), (6, 3), (3, 4))], {2: 1}),
                 ((12, 12), (6, 6, 1), [dict(r=1, c=1, h=3, w=3, f=2),
                  dict(r=0, c=6, h=2, w=3, f=3), dict(r=5, c=1, h=3, w=2, f=2)], {2: 1, 3: -1})]
    for args in malformed:
        try:
            _trace(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ray geometry accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), scene_geometries=len(geometry),
                field_shapes=len(shapes), footprints=len(footprints), cargo_counts=sorted(counts),
                max_turn_counts=sorted(depths), patterned_pairs=patterned,
                repeated_hit_pairs=repeat_hits, palette_sets=len(palettes),
                verification="generator-owned checks only")


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
