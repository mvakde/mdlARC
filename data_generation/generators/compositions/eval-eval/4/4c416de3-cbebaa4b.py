"""Complementary corner rivets: mutually repair and assemble rectangular frames.

Only standard-library modules and NumPy are used. Inference has no access to
random state, construction plans, training labels, or stored examples.
"""
from itertools import combinations
import json
import numpy as np

DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
HOST_SEEDS = (0, 581448278, 914961609, 37803325, 274174468, 970057511,
              12471480, 150815130, 371006663, 429829780, 945463701,
              266371818, 222950978, 282225935, 151239630, 131449348,
              288208638, 891790300, 731664278, 272260374, 418743325,
              812153482, 181920316, 849636184, 168227591, 495523290,
              959442503, 18373249, 851313623, 183340522, 499683249,
              410826404, 719488953, 740282887, 364893856, 730180550,
              74219217, 981177512, 602952333, 971921990, 104755605,
              425104120)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a.astype(np.int8, copy=True)


def _neg(points):
    return frozenset((-r, -c) for r, c in points)


def _shift(points, d):
    return {(r + d[0], c + d[1]) for r, c in points}


def _box(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _rect(box):
    t, l, b, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _outline(box):
    t, l, b, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _corners(box):
    t, l, b, r = box
    return ((t, l), (t, r), (b, l), (b, r))


def _center(box):
    t, l, b, r = box
    return (t + b) // 2, (l + r) // 2


def _at(corner, q, mask):
    dy, dx = DIRS[q]
    return {(corner[0] + dy * u, corner[1] + dx * v) for u, v in mask}


def _components(points):
    left = set(points)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        found, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in left:
                        left.remove(p)
                        found.add(p)
                        todo.append(p)
        result.append(found)
    return result


def _read_input(grid):
    """Recover frame geometry, dot, master motif, and every fragment."""
    a = _grid(grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert np.count_nonzero(counts == counts[bg]) == 1, "background tie"
    frames = []
    for color in range(10):
        if color == bg:
            continue
        for comp in _components(map(tuple, np.argwhere(a == color).tolist())):
            box = _box(comp)
            t, l, b, r = box
            if b - t < 4 or r - l < 4:
                continue
            outline = _outline(box)
            corners = set(_corners(box))
            if comp <= outline and outline - corners <= comp:
                frames.append(dict(box=box, color=color))
    frames.sort(key=lambda p: p['box'])
    assert 3 <= len(frames) <= 6, "expected three to six readable frames"
    # The dots and ornaments cannot create a large rectangular perimeter.
    frame_cells = set()
    for i, f in enumerate(frames):
        region = _rect(f['box'])
        assert all(not region & _rect(g['box']) for g in frames[:i]), "input frames overlap"
        frame_cells |= {p for p in _outline(f['box']) if a[p] == f['color']}
    centers = [(i, _center(f['box'])) for i, f in enumerate(frames)
               if a[_center(f['box'])] != bg]
    assert len(centers) == 1, "one interior anchor dot is required"
    anchor, dot = centers[0]
    dot_color = int(a[dot])
    assert counts[dot_color] == 1, "dot hue must be reserved"
    rest = set(map(tuple, np.argwhere(a != bg).tolist())) - frame_cells - {dot}
    assert rest
    hues = {int(a[p]) for p in rest}
    assert len(hues) == 1, "rivet hue must be common"
    marker = hues.pop()
    assert marker not in {f['color'] for f in frames} | {bg, dot_color}
    all_corners = [(i, q, p) for i, f in enumerate(frames)
                   for q, p in enumerate(_corners(f['box']))]
    groups = {}
    for p in sorted(rest):
        distances = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                     for _, _, c in all_corners]
        nearest = min(distances)
        assert distances.count(nearest) == 1, "fragment has two equally near corners"
        i, q, c = all_corners[distances.index(nearest)]
        dy, dx = DIRS[q]
        u, v = (p[0] - c[0]) * dy, (p[1] - c[1]) * dx
        assert max(abs(u), abs(v)) <= 2 and ((u == v == 0) or (u != 0 and v != 0))
        groups.setdefault((i, q), set()).add((u, v))
    ports = [dict(frame=i, q=q, corner=_corners(frames[i]['box'])[q],
                  mask=frozenset(mask)) for (i, q), mask in sorted(groups.items())]
    maximal = {p['mask'] for p in ports
               if all(other['mask'] <= p['mask'] for other in ports)}
    assert len(maximal) == 1, "no unique intact motif"
    motif = maximal.pop()
    assert 5 <= len(motif) <= 13 and (0, 0) in motif and _neg(motif) == motif
    full = [p for p in ports if p['mask'] == motif]
    assert len(full) == 1 and full[0]['frame'] == anchor
    partials = [p for p in ports if p['mask'] != motif]
    assert len(partials) == 2 * (len(frames) - 1)
    radius = max(max(abs(u), abs(v)) for u, v in motif)
    assert all(min(f['box'][2] - f['box'][0], f['box'][3] - f['box'][1]) >= 2 * radius + 2
               for f in frames), "corner neighborhoods would overlap"
    for p in ports:
        completed = _at(p['corner'], p['q'], motif)
        assert all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in completed)
        for r, c in completed:
            ds = [(r - y) ** 2 + (c - x) ** 2 for _, _, (y, x) in all_corners]
            own = next(k for k, (i, q, _) in enumerate(all_corners)
                       if (i, q) == (p['frame'], p['q']))
            assert ds.count(min(ds)) == 1 and ds[own] == min(ds)
            assert a[r, c] in (bg, marker, frames[p['frame']]['color'])
    return dict(grid=a, background=bg, marker=marker, frames=frames,
                anchor=anchor, dot=dot, dot_color=dot_color, motif=motif,
                master=full[0], ports=partials, radius=radius)


def _infer(grid):
    """Complement pairing fixes the tree; coincident corners fix translations."""
    info = _read_input(grid)
    a, frames, ports, motif = (info[k] for k in ('grid', 'frames', 'ports', 'motif'))
    mates = {}
    for i, p in enumerate(ports):
        wanted = _neg(motif - p['mask'])
        options = [j for j, other in enumerate(ports)
                   if other['frame'] != p['frame'] and other['q'] == (p['q'] ^ 3)
                   and other['mask'] == wanted]
        assert len(options) == 1, "missing or ambiguous complementary mate"
        mates[i] = options[0]
    assert all(mates[mates[i]] == i for i in mates)
    edges = [(ports[i], ports[j]) for i, j in mates.items() if i < j]
    assert len(edges) == len(frames) - 1
    graph = [[] for _ in frames]
    for p, q in edges:
        d = p['corner'][0] - q['corner'][0], p['corner'][1] - q['corner'][1]
        graph[p['frame']].append((q['frame'], d))
        graph[q['frame']].append((p['frame'], (-d[0], -d[1])))
    positions = {info['anchor']: (0, 0)}
    todo = [info['anchor']]
    while todo:
        i = todo.pop()
        for j, d in graph[i]:
            proposed = positions[i][0] + d[0], positions[i][1] + d[1]
            if j in positions:
                assert positions[j] == proposed, "inconsistent cycle"
            else:
                positions[j] = proposed
                todo.append(j)
    assert len(positions) == len(frames), "assembly is disconnected"
    placed_boxes = []
    for i, f in enumerate(frames):
        t, l, b, r = f['box']
        dy, dx = positions[i]
        placed_boxes.append((t + dy, l + dx, b + dy, r + dx))
    joints = {}
    for p, q in edges:
        cp = next(iter(_shift({p['corner']}, positions[p['frame']])))
        cq = next(iter(_shift({q['corner']}, positions[q['frame']])))
        assert cp == cq and cp not in joints, "multiple joints at one corner"
        full = _at(cp, p['q'], motif)
        assert full == _at(cq, q['q'], motif)
        survivors_p = _shift(_at(p['corner'], p['q'], p['mask']), positions[p['frame']])
        survivors_q = _shift(_at(q['corner'], q['q'], q['mask']), positions[q['frame']])
        assert not survivors_p & survivors_q and survivors_p | survivors_q == full
        joints[cp] = (frozenset((p['frame'], q['frame'])), full)
    for i, j in combinations(range(len(frames)), 2):
        overlap = _rect(placed_boxes[i]) & _rect(placed_boxes[j])
        allowed = {c for c, (owners, _) in joints.items() if owners == frozenset((i, j))}
        assert overlap == allowed, "frame interiors or unrelated outlines collide"
    out = np.full(a.shape, info['background'], dtype=np.int8)
    for i, f in enumerate(frames):
        for r, c in _outline(placed_boxes[i]):
            assert 0 <= r < a.shape[0] and 0 <= c < a.shape[1], "assembly outside canvas"
            out[r, c] = f['color']
    out[info['dot']] = info['dot_color']
    master = info['master']
    master_cells = _at(master['corner'], master['q'], motif)
    ornaments = [master_cells] + [full for _, full in joints.values()]
    seen = set()
    for cells in ornaments:
        assert not cells & seen, "rivet motifs collide"
        seen |= cells
        for r, c in cells:
            assert 0 <= r < a.shape[0] and 0 <= c < a.shape[1]
            assert (r, c) != info['dot']
            out[r, c] = info['marker']
    # Rivets may overwrite only their owners' corner cells, not another frame.
    for cp, (owners, cells) in joints.items():
        for i in range(len(frames)):
            assert cells & _outline(placed_boxes[i]) <= ({cp} if i in owners else set())
    assert all(not master_cells & _outline(placed_boxes[i]) for i in range(len(frames))
               if i != info['anchor'])
    info.update(positions=positions, edges=edges, graph=graph, placed_boxes=placed_boxes,
                joints=joints, prediction=out)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique complementary-rivet assembly using input only."""
    return _infer(input_grid)['prediction'].copy()


def _count_only_alternative(info):
    """A concrete wrong order permitted by direction and cardinality alone.

    The first demonstration is a monotone four-frame chain with an endpoint
    anchor. Swapping its two internal frames preserves sizes, port degrees,
    corner directions, split cardinalities, bounds, and collision freedom.
    Only the complementary masks specify their correct order.
    """
    anchor, graph = info['anchor'], info['graph']
    assert len(graph) == 4 and len(graph[anchor]) == 1
    order, previous = [anchor], None
    while len(order) < 4:
        choices = [j for j, _ in graph[order[-1]] if j != previous]
        assert len(choices) == 1
        previous = order[-1]
        order.append(choices[0])
    order[1], order[2] = order[2], order[1]
    q = next(p['q'] for p in info['ports'] if p['frame'] == anchor)
    by_corner = {(p['frame'], p['q']): p for p in info['ports']}
    assert len({f['color'] for f in info['frames']}) == 4
    positions = {anchor: (0, 0)}
    for i, j in zip(order, order[1:]):
        p, s = by_corner[i, q], by_corner[j, q ^ 3]
        assert len(p['mask']) != len(s['mask'])
        positions[j] = (positions[i][0] + p['corner'][0] - s['corner'][0],
                        positions[i][1] + p['corner'][1] - s['corner'][1])
    out = np.full(info['grid'].shape, info['background'], dtype=np.int8)
    for i, f in enumerate(info['frames']):
        for r, c in _shift(_outline(f['box']), positions[i]):
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            out[r, c] = f['color']
    out[info['dot']] = info['dot_color']
    fulls = [_at(info['master']['corner'], info['master']['q'], info['motif'])]
    for i in order[:-1]:
        p = by_corner[i, q]
        fulls.append(_shift(_at(p['corner'], q, info['motif']), positions[i]))
    for cells in fulls:
        for r, c in cells:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            out[r, c] = info['marker']
    return out


def _rotation_alternative(info):
    """Replace axis-reflected motifs by a consistent quarter-turn convention."""
    out = np.full(info['grid'].shape, info['background'], dtype=np.int8)
    for i, f in enumerate(info['frames']):
        for cell in _shift(_outline(f['box']), info['positions'][i]):
            out[cell] = f['color']
    out[info['dot']] = info['dot_color']
    master = info['master']
    fulls = [_at(master['corner'], master['q'], info['motif'])]
    for p, _ in info['edges']:
        motif = info['motif']
        if (p['q'] in (1, 2)) != (master['q'] in (1, 2)):
            motif = frozenset((v, u) for u, v in motif)
        fulls.append(_shift(_at(p['corner'], p['q'], motif), info['positions'][p['frame']]))
    for cells in fulls:
        for cell in cells:
            out[cell] = info['marker']
    return out


def _sample_motif(rng, radius, asymmetric):
    # Central inversion orbits; axis arms stay untouched except the origin.
    orbits = [frozenset(((u, v), (-u, -v)))
              for u in range(1, radius + 1)
              for v in range(-radius, radius + 1) if v]
    for _ in range(100):
        n = 2 if radius == 1 else int(rng.integers(3, 7))
        chosen = rng.choice(len(orbits), n, replace=False)
        motif = frozenset({(0, 0)}.union(*(orbits[int(i)] for i in chosen)))
        if not asymmetric or motif != frozenset((v, u) for u, v in motif):
            return motif
    raise RuntimeError("could not sample an asymmetric motif")


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _layout(rng, count, radius, branch, straight=False, extended=False):
    minimum = 2 * radius + 3
    maximum = minimum + (3 if extended else (2 if count <= 4 else 1))
    for _ in range(100):
        sizes = [(int(rng.integers(minimum, maximum + 1)),
                  int(rng.integers(minimum, maximum + 1))) for i in range(count)]
        if extended:
            i, axis = int(rng.integers(count)), int(rng.integers(2))
            sizes[i] = (maximum, sizes[i][1]) if axis == 0 else (sizes[i][0], maximum)
        boxes = [(0, 0, sizes[0][0] - 1, sizes[0][1] - 1)]
        used, edges = [set()], []
        direction = int(rng.integers(4))
        for j in range(1, count):
            options = [(i, q) for i in range(j) for q in range(4) if q not in used[i]]
            if straight:
                options = [(j - 1, direction)]
            rng.shuffle(options)
            if not branch:
                options.sort(key=lambda z: z[0] != j - 1)
            for i, q in options:
                y, x = _corners(boxes[i])[q]
                h, w = sizes[j]
                other = q ^ 3
                top = y if other < 2 else y - h + 1
                left = x if other % 2 == 0 else x - w + 1
                new = (top, left, top + h - 1, left + w - 1)
                region = _rect(new)
                if any(region & _rect(b) != ({(y, x)} if k == i else set())
                       for k, b in enumerate(boxes)):
                    continue
                all_cells = set().union(*(_rect(b) for b in boxes), region)
                t, l, b, r = _box(all_cells)
                if b - t + 1 + 2 * radius > 30 or r - l + 1 + 2 * radius > 30:
                    continue
                boxes.append(new)
                used[i].add(q)
                used.append({other})
                edges.append((i, q, j, other))
                break
            else:
                break
        if len(boxes) == count and (not branch or max(map(len, used)) >= 3):
            return boxes, used, edges
    raise RuntimeError("bounded frame-tree construction exhausted")


def _scatter(rng, shape, boxes, anchor, radius):
    """Random rectangle packing; enlarged boxes reserve readable corner halos."""
    h, w = shape
    def halo(box):
        t, l, b, r = box
        return t - radius, l - radius, b + radius, r + radius
    def separate(a, b):
        return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]
    for _ in range(32):
        placements = {anchor: (0, 0)}
        occupied = [halo(boxes[anchor])]
        order = [i for i in range(len(boxes)) if i != anchor]
        rng.shuffle(order)
        order.sort(key=lambda i: -(boxes[i][2] - boxes[i][0] + 1 + 2 * radius)
                                  * (boxes[i][3] - boxes[i][1] + 1 + 2 * radius))
        for i in order:
            t, l, b, r = boxes[i]
            rh, rw = b - t + 1, r - l + 1
            candidates = [(y - t, x - l)
                          for y in range(radius, h - rh - radius + 1)
                          for x in range(radius, w - rw - radius + 1)]
            rng.shuffle(candidates)
            for d in candidates:
                if d == (0, 0):
                    continue
                ht, hl, hb, hr = halo(boxes[i])
                moved = ht + d[0], hl + d[1], hb + d[0], hr + d[1]
                if all(separate(moved, old) for old in occupied):
                    placements[i] = d
                    occupied.append(moved)
                    break
            else:
                break
        if len(placements) == len(boxes):
            return placements
    return None


def _make_pair(rng, count, radius, branch, roles, witness=False, key_witness=False, extended=False):
    bg, marker, dot_color = roles
    for _ in range(100):
        motif = _sample_motif(rng, radius, witness)
        try:
            boxes, used, edges = _layout(rng, count, radius, branch,
                                        straight=key_witness, extended=extended)
        except RuntimeError:
            continue
        eligible = [i for i in range(count) if len(used[i]) < 4]
        if key_witness:
            eligible = [i for i in eligible if len(used[i]) == 1]
        anchor = int(_choose(rng, eligible))
        masters = [q for q in range(4) if q not in used[anchor]
                   and any(q ^ p in (1, 2) for p in used[anchor])]
        if not masters:
            continue
        master_q = int(_choose(rng, masters))
        if key_witness:
            masks = list(combinations(sorted(motif), len(motif) // 2))
            rng.shuffle(masks)
            selected = [frozenset(m) for m in masks[:len(edges)]]
        else:
            selected, seen_masks = [], set()
            cells = sorted(motif)
            for _ in range(200):
                if len(selected) == len(edges):
                    break
                keep = int(rng.integers(1, len(motif)))
                indexes = rng.choice(len(cells), keep, replace=False)
                mask = frozenset(cells[int(i)] for i in indexes)
                mate = _neg(motif - mask)
                if mask not in seen_masks and mate not in seen_masks:
                    selected.append(mask)
                    seen_masks.update((mask, mate))
            if len(selected) != len(edges):
                continue
        ornaments = {(anchor, master_q): motif}
        for (i, q, j, s), mask in zip(edges, selected):
            ornaments[i, q] = mask
            ornaments[j, s] = _neg(motif - mask)
        palette = [c for c in range(10) if c not in roles]
        rng.shuffle(palette)
        palette_size = int(rng.integers(2, min(len(palette), count) + 1))
        if key_witness:
            palette_size = count
        colors = [palette[i % palette_size] for i in range(count)]
        rng.shuffle(colors)
        all_cells = set().union(*(_rect(b) for b in boxes))
        for (i, q), mask in ornaments.items():
            all_cells |= _at(_corners(boxes[i])[q], q, motif)
        t, l, b, r = _box(all_cells)
        extent_h, extent_w = b - t + 1, r - l + 1
        h = min(30, max(22, extent_h + int(rng.integers(2, 8))))
        w = min(30, max(22, extent_w + int(rng.integers(2, 8))))
        # The reference anchor must allow scattering as well as final assembly.
        if (count >= 4 and radius == 2) or count >= 5:
            h = w = 30
        offset = int(rng.integers(h - extent_h + 1)) - t, int(rng.integers(w - extent_w + 1)) - l
        boxes = [(a + offset[0], c + offset[1], d + offset[0], e + offset[1])
                 for a, c, d, e in boxes]
        # Scattered frames include their full corner neighborhoods in bounds.
        at, al, ab, ar = boxes[anchor]
        if at < radius or al < radius or ab + radius >= h or ar + radius >= w:
            continue
        scattering = _scatter(rng, (h, w), boxes, anchor, radius)
        if scattering is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        for i, box in enumerate(boxes):
            for p in _outline(box):
                out[p] = colors[i]
            for p in _shift(_outline(box), scattering[i]):
                inp[p] = colors[i]
        dot = _center(boxes[anchor])
        inp[dot] = out[dot] = dot_color
        for (i, q), mask in ornaments.items():
            corner = _corners(boxes[i])[q]
            for p in _shift(_at(corner, q, mask), scattering[i]):
                inp[p] = marker
            for p in _at(corner, q, motif):
                out[p] = marker
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        try:
            info = _infer(inp)
            assert np.array_equal(info['prediction'], out)
            assert all(d != (0, 0) for i, d in info['positions'].items() if i != info['anchor'])
            if witness:
                assert info['motif'] != frozenset((v, u) for u, v in info['motif'])
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError("bounded sampling did not find a readable assembly")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    # Two demonstrations suffice even with the minimum legal training count:
    # an ordering witness with interchangeable middle pieces under count-only
    # matching, then an asymmetric adjacent-corner witness on a branch.
    regime = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train':
                n = 4 if index < 2 else int(rng.integers(3, 5))
                radius = 1 if index == 0 else (2 if index == 1 else int(rng.integers(1, 3)))
                branch = index == 1 or (n == 4 and rng.random() < .6)
                if index == 0:
                    branch = False
            else:
                n = int(rng.integers(5, 7)) if regime == 0 else 4
                radius = 1 if regime == 0 else 2
                branch = True
            for _ in range(30):
                pair = _make_pair(rng, n, radius, branch, roles,
                                  witness=split == 'train' and index == 1,
                                  key_witness=split == 'train' and index == 0,
                                  extended=split == 'test' and regime == 1)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, roles, records = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _infer(inp)
            assert np.array_equal(info['prediction'], out), "answer violates complementary docking"
            roles.add((info['background'], info['marker'], info['dot_color']))
            assert info['positions'][info['anchor']] == (0, 0)
            assert all(d != (0, 0) for i, d in info['positions'].items() if i != info['anchor'])
            assert out[info['dot']] == info['dot_color']
            assert np.count_nonzero(inp == info['marker']) == np.count_nonzero(out == info['marker']), "rivet conservation failed"
            assert np.count_nonzero(inp != info['background']) - np.count_nonzero(out != info['background']) == len(info['edges'])
            assert set(np.unique(inp)) == set(np.unique(out))
            # A-only counterrule: reflect/complete every fragment in place.
            a_only = inp.copy()
            for p in info['ports']:
                for cell in _at(p['corner'], p['q'], info['motif']):
                    a_only[cell] = info['marker']
            assert not np.array_equal(a_only, out)
            # B-only identical-signature docking cannot pair this complement
            # graph: every actual joint has unequal fragment cardinalities.
            assert all(len(p['mask']) != len(q['mask']) for p, q in info['edges'])
            records[split].append(info)
    assert len(roles) == 1
    first, second = records['train'][:2]
    assert not np.array_equal(_count_only_alternative(first), first['prediction'])
    assert second['motif'] != frozenset((v, u) for u, v in second['motif'])
    assert any(second['master']['q'] ^ p['q'] in (1, 2)
               for p in second['ports'] if p['frame'] == second['anchor'])
    assert not np.array_equal(_rotation_alternative(second), second['prediction'])
    assert max(len(g) for g in second['graph']) >= 3
    # Equal-area alternatives are required in the ordering witness. Other
    # demonstrations can use independently sampled complementary split sizes.
    assert any(len(p['mask']) == len(q['mask']) and p['mask'] != q['mask']
               for p, q in combinations(first['ports'], 2))
    maximum_train_count = max(len(i['frames']) for i in records['train'])
    maximum_train_side = max(max(f['box'][2] - f['box'][0] + 1,
                                 f['box'][3] - f['box'][1] + 1)
                             for i in records['train'] for f in i['frames'])
    for i in records['test']:
        maximum_side = max(max(f['box'][2] - f['box'][0] + 1,
                               f['box'][3] - f['box'][1] + 1) for f in i['frames'])
        assert len(i['frames']) > maximum_train_count or maximum_side > maximum_train_side


def _signature(puzzle):
    return tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Bounded semantic checks and structural coverage on 42 specified seeds."""
    shapes, motifs, boxes, counts, degrees, palettes, signatures, offsets = (set() for _ in range(8))
    reused = asymmetric = total = 0
    fragment_sizes = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                info = _infer(p['input'].copy())
                assert np.array_equal(solve(p['input'].tolist()), p['output'])
                shapes.add(p['input'].shape)
                motifs.add(tuple(sorted(info['motif'])))
                counts.add(len(info['frames']))
                degrees.update(map(len, info['graph']))
                colors = [f['color'] for f in info['frames']]
                reused += len(set(colors)) < len(colors)
                asymmetric += info['motif'] != frozenset((v, u) for u, v in info['motif'])
                palettes.add((info['background'], info['marker'], info['dot_color']))
                offsets.update(info['positions'].values())
                fragment_sizes.update(len(p['mask']) for p in info['ports'])
                for t, l, b, r in (f['box'] for f in info['frames']):
                    boxes.add((b - t + 1, r - l + 1))
                total += 1
    assert len(signatures) == 42 and len(motifs) >= 30
    assert len(shapes) >= 20 and len(boxes) >= 12
    assert counts >= {3, 4, 5, 6} and degrees >= {1, 2, 3}
    assert len(palettes) >= 20 and len(offsets) >= 100
    assert reused > 0 and asymmetric >= 42
    assert len(fragment_sizes) >= 8 and 1 in fragment_sizes
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(31415, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(0, 2, 1)
    info = _infer(bad['test'][0]['input'])
    cell = next(iter(next(iter(info['joints'].values()))[1]))
    bad['test'][0]['output'][cell] = info['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased joint accepted")
    broken = generate_puzzle(0, 2, 1)['train'][0]['input'].copy()
    info = _infer(broken)
    port = info['ports'][0]
    cell = next(p for p in _at(port['corner'], port['q'], port['mask'])
                if p != port['corner'])
    broken[cell] = info['background']
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing complementary key pixel accepted")
    return {'seeds': len(HOST_SEEDS), 'pairs': total, 'motifs': len(motifs),
            'canvas_shapes': len(shapes), 'frame_shapes': len(boxes),
            'piece_counts': sorted(counts), 'degrees': sorted(degrees),
            'role_palettes': len(palettes), 'reused_payload_pairs': int(reused),
            'asymmetric_pairs': int(asymmetric), 'fragment_sizes': sorted(fragment_sizes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
