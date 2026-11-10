"""Relational jigsaws with two-hop color locks across rigid-piece seams.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

from collections import deque
from math import isqrt

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Integer colors required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells, directions=D8):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _tight(a, cells):
    cells = list(cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    r1 = max(r for r, c in cells) + 1
    c1 = max(c for r, c in cells) + 1
    p = np.full((r1-r0, c1-c0), -1, dtype=np.int8)
    for r, c in cells:
        p[r-r0, c-c0] = a[r, c]
    return p


def _port(piece, body, eye, line):
    """Read one open line: eye-side endpoint to boundary-side endpoint."""
    cells = {tuple(map(int, p)) for p in np.argwhere(piece == line)}
    if not cells:
        return None
    _require(len(cells) >= 2, "A port must contain at least two rail pixels")
    graph = {p: [(p[0]+dr, p[1]+dc) for dr, dc in D4
                 if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    _require(len(_components(cells, D4)) == 1, "Broken port")
    _require(all(1 <= len(adj) <= 2 for adj in graph.values()), "Branched port")
    ends = [p for p in cells if len(graph[p]) == 1]
    _require(len(ends) == 2, "Port must be open")
    attached = [p for p in cells if abs(p[0]-eye[0])+abs(p[1]-eye[1]) == 1]
    _require(len(attached) == 1 and attached[0] in ends, "Eye must select one line end")
    tip = next(p for p in ends if p != attached[0])
    previous = graph[tip][0]
    target = 2*tip[0]-previous[0], 2*tip[1]-previous[1]
    h, w = piece.shape
    _require(not (0 <= target[0] < h and 0 <= target[1] < w
                  and piece[target] >= 0), "Port must point through a fragment edge")
    return target


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    _require(int(np.sum(counts == counts.max())) == 1, "Background must be uniquely modal")
    bg = int(colors[np.argmax(counts)])
    marker_candidates = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    _require(len(marker_candidates) == 1, "A unique singleton corner marker is required")
    marker = marker_candidates[0]
    components = _components(map(tuple, np.argwhere(a != bg)))
    _require(3 <= len(components) <= 6, "Require three to six separated fragments")
    pieces = [_tight(a, cells) for cells in components]
    bodies = []
    for p in pieces:
        vals, nums = np.unique(p[(p >= 0) & (p != marker)], return_counts=True)
        body = int(vals[np.argmax(nums)])
        _require(np.sum(p == body) > np.sum(p >= 0)/2, "Body must occupy a strict majority")
        bodies.append(body)
    _require(len(set(bodies)) == len(bodies), "Body identities must be distinct")
    extras = set(map(int, colors)) - {bg, marker} - set(bodies)
    _require(len(extras) == 1, "One neutral port color is required")
    line = extras.pop()
    relations, records = {}, []
    for p, body in zip(pieces, bodies):
        eye_cells = [(int(r), int(c)) for r, c in np.argwhere(
            (p >= 0) & (p != body) & (p != marker) & (p != line))]
        _require(len(eye_cells) == 1, "Each fragment must expose exactly one different-color eye")
        eye = eye_cells[0]
        destination = int(p[eye])
        _require(destination in bodies and destination != body, "Eye must name another body")
        relations[body] = destination
        records.append(dict(array=p, body=body, eye=eye,
                            target=_port(p, body, eye, line)))
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = isqrt(area)
    _require(side*side == area, "Fragment area must form a square")
    _require(sum(np.sum(p == marker) for p in pieces) == 1, "One attached marker required")
    _require(any(r['target'] is not None for r in records), "At least one seam lock required")
    return dict(background=bg, marker=marker, line=line, side=side,
                records=records, relations=relations)


def _orientations(piece, rotations=True):
    seen = set()
    for k in range(4 if rotations else 1):
        p = np.rot90(piece, k).copy()
        if _key(p) not in seen:
            seen.add(_key(p))
            yield p


def _pack(facts, *, use_locks=True, direct=False, rotations=True, plain_body=True, limit=2):
    """Exhaustive input-derived exact cover with seam constraints.

    Distinct *final colored answers* are counted. Rotating a donor-only symmetric
    support may change its erased eye's location without changing the answer.
    """
    side, marker, line = facts['side'], facts['marker'], facts['line']
    mapping = facts['relations']
    candidates = []
    anchor = None
    for i, rec in enumerate(facts['records']):
        body = rec['body']
        marked = bool(np.any(rec['array'] == marker))
        if marked:
            anchor = i
        options, signatures = [], set()
        for p in _orientations(rec['array'], rotations):
            h, w = p.shape
            if h > side or w > side:
                continue
            eye = tuple(map(int, np.argwhere((p >= 0) & (p != body)
                         & (p != marker) & (p != line))[0]))
            target = _port(p, body, eye, line)
            locked = target is not None
            replacement = mapping[mapping[body]] if locked else body
            required = mapping[body] if direct else mapping[mapping[body]]
            if marked:
                if p[0, 0] != marker:
                    continue
                offsets = [(0, 0)]
            else:
                offsets = [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
            local = [(int(r), int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0)]
            for row, col in offsets:
                cells = tuple(((row+r)*side+col+c, color) for r, c, color in local)
                mask = sum(1 << pos for pos, color in cells)
                lock = None
                if locked and use_locks:
                    tr, tc = row+target[0], col+target[1]
                    if not (0 <= tr < side and 0 <= tc < side):
                        continue
                    tpos = tr*side+tc
                    if mask & (1 << tpos):
                        continue
                    lock = (tpos, required)
                epos = (row+eye[0])*side+col+eye[1]
                painted = tuple((pos, replacement if pos == epos else color)
                                for pos, color in cells)
                signature = (mask, painted, lock)
                if signature not in signatures:
                    signatures.add(signature)
                    options.append((mask, cells, painted, lock))
        if not options:
            return []
        candidates.append(options)
    _require(anchor is not None, "Marker must belong to one fragment")
    full = (1 << (side*side)) - 1
    original = [-1]*(side*side)
    owners = [-1]*(side*side)
    placed = [None]*len(candidates)
    solutions, seen = [], set()

    def search(occupied, remaining, requirements):
        if len(solutions) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            out = np.empty((side, side), dtype=np.int8)
            for option in placed:
                for pos, color in option[2]:
                    out.flat[pos] = color
            signature = out.tobytes()
            if signature not in seen:
                seen.add(signature)
                solutions.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        order = (anchor,) if anchor in remaining else remaining
        for i in order:
            body = facts['records'][i]['body']
            rest = tuple(j for j in remaining if j != i)
            for option in candidates[i]:
                mask, cells, painted, lock = option
                if mask & occupied or (anchor not in remaining and not mask & first):
                    continue
                if any(pos in requirements and
                       (requirements[pos] != color or (plain_body and requirements[pos] != body))
                       for pos, color in cells):
                    continue
                req = requirements
                if lock is not None:
                    pos, color = lock
                    if original[pos] >= 0 and (original[pos] != color or
                                               (plain_body and owners[pos] != color)):
                        continue
                    if pos in requirements and requirements[pos] != color:
                        continue
                    req = dict(requirements)
                    req[pos] = color
                for pos, color in cells:
                    original[pos] = color
                    owners[pos] = body
                placed[i] = option
                search(occupied | mask, rest, req)
                for pos, color in cells:
                    original[pos] = -1
                    owners[pos] = -1
                placed[i] = None
                if len(solutions) >= limit:
                    return
    search(0, tuple(range(len(candidates))), {})
    return solutions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the visible fragments and relations, without any generation state."""
    answers = _pack(_read_input(input_grid))
    if len(answers) != 1:
        raise ValueError("Input must have exactly one final colored locked assembly")
    return answers[0].copy()


def _frame_partition(rng, side, count):
    """Two interchangeable asymmetric inserts, plus optional smaller cavities."""
    labels = np.zeros((side, side), dtype=np.int8)
    h, w = 4, 3
    if side >= 11 and rng.random() < .5:
        h = 5
    r = int(rng.integers(3, side-h))
    left = int(rng.integers(1, max(2, side//2-w+1)))
    right = int(rng.integers(max(left+w+1, side//2), side-w+1))
    shape = np.ones((h, w), dtype=bool)
    shape[0, -1] = False
    if h == 5 and rng.random() < .5:
        shape[1, -1] = False
    for label, col in ((1, left), (2, right)):
        labels[r:r+h, col:col+w][shape] = label
    # Additional cavities have different areas, so only the twin exchange is
    # the intended geometry-only ambiguity. Their positions are resampled.
    for label in range(3, count):
        hh, ww = 2, label
        options = []
        for rr in range(1, side-hh):
            for cc in range(1, side-ww):
                box = labels[max(0, rr-1):rr+hh+1, max(0, cc-1):cc+ww+1]
                if not np.any(box):
                    options.append((rr, cc))
        if not options:
            return None
        rr, cc = options[int(rng.integers(len(options)))]
        labels[rr:rr+hh, cc:cc+ww] = label
    return labels


def _bands_partition(rng, side, count):
    # Monotone, stepped seams keep each region connected while changing width
    # row by row. Straight bands remain part of the spectrum.
    cuts = sorted(map(int, rng.choice(np.arange(2, side-1), count-1, replace=False)))
    if min(np.diff([0]+cuts+[side])) < 2:
        return None
    labels = np.empty((side, side), dtype=np.int8)
    for r in range(side):
        if r and rng.random() < .7:
            j = int(rng.integers(count-1))
            proposal = cuts.copy()
            proposal[j] += int(rng.choice([-1, 1]))
            if min(np.diff([0]+proposal+[side])) >= 2:
                cuts = proposal
        edges = [0]+cuts+[side]
        for i in range(count):
            labels[r, edges[i]:edges[i+1]] = i
    return np.rot90(labels, int(rng.integers(4))).copy()


def _growth_partition(rng, side, count):
    labels = np.full((side, side), -1, dtype=np.int8)
    points = [0]+list(map(int, rng.choice(np.arange(1, side*side), count-1, replace=False)))
    weights = rng.uniform(.5, 2., count)
    frontier = [set() for _ in range(count)]
    sizes = [1]*count
    for i, pos in enumerate(points):
        labels.flat[pos] = i
    def neighbors(pos):
        r, c = divmod(pos, side)
        return [rr*side+cc for dr, dc in D4
                if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]
    for i, pos in enumerate(points):
        frontier[i].update(q for q in neighbors(pos) if labels.flat[q] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random()))
        choices = sorted(frontier[i])
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(pos)
        frontier[i].update(q for q in neighbors(pos) if labels.flat[q] < 0)
    return labels


def _gate_options(labels, body, target, reserved):
    side = len(labels)
    max_rail = min(5, (int(np.sum(labels == body))-1)//2-1-int(body == labels[0, 0]))
    options = []
    # Enumerate short, nonbranching straight and once-bent ports. The first
    # two rail pixels are straight and point toward the intended seam.
    for r, c in map(tuple, np.argwhere(labels == body)):
        for dr, dc in D4:
            outside = r+dr, c+dc
            inner = r-dr, c-dc
            if not (0 <= outside[0] < side and 0 <= outside[1] < side
                    and labels[outside] == target):
                continue
            if not (0 <= inner[0] < side and 0 <= inner[1] < side
                    and labels[inner] == body):
                continue
            path = [(int(r), int(c)), (int(inner[0]), int(inner[1]))]
            if any(p == (0, 0) or p in reserved for p in path):
                continue
            queue = deque([(path, (-dr, -dc), False)])
            while queue:
                rails, direction, bent = queue.popleft()
                end = rails[-1]
                dirs = [direction] + ([] if bent else [d for d in D4 if d[0]*direction[0]+d[1]*direction[1] == 0])
                for step in dirs:
                    eye = end[0]+step[0], end[1]+step[1]
                    if not (0 <= eye[0] < side and 0 <= eye[1] < side
                            and labels[eye] == body and eye != (0, 0)
                            and eye not in reserved and eye not in rails):
                        continue
                    if sum(abs(eye[0]-p[0])+abs(eye[1]-p[1]) == 1 for p in rails) != 1:
                        continue
                    options.append((tuple(rails), eye, outside))
                    if len(rails) < max_rail:
                        queue.append((rails+[eye], step, bent or step != direction))
    return options


def _annotate(rng, labels, count, palette, witness):
    bg, marker, line = palette[:3]
    colors = palette[3:3+count]
    if witness:
        mapping = [1, 2, 0]+[int(rng.integers(3)) for _ in range(count-3)]
        active = [1, 0]
    else:
        if rng.random() < .65:
            order = list(map(int, rng.permutation(count)))
            mapping = [0]*count
            for a, b in zip(order, order[1:]+order[:1]):
                mapping[a] = b
        else:
            mapping = [int(rng.choice([j for j in range(count) if i != j])) for i in range(count)]
        available = [i for i in range(count) if mapping[mapping[i]] != i
                     and _gate_options(labels, i, mapping[mapping[i]], set())]
        if len(available) < 2:
            return None
        active = list(map(int, rng.choice(available,
                      int(rng.integers(2, min(count-1, len(available))+1)), replace=False)))
    original = np.empty(labels.shape, dtype=np.int8)
    for i, color in enumerate(colors):
        original[labels == i] = color
    original[0, 0] = marker
    reserved, used, eyes = set(), {(0, 0)}, {}
    if not witness:
        rng.shuffle(active)
    for i in active:
        options = _gate_options(labels, i, mapping[mapping[i]], reserved)
        options = [o for o in options if o[2] not in used]
        if witness:
            if i == 1:
                # Put this selected eye on its insert's top edge. The enclosing
                # port will face the corresponding location on the other twin.
                top = int(np.min(np.argwhere(labels == 1)[:, 0]))
                options = [o for o in options if o[1][0] == top]
            elif i == 0:
                a = np.min(np.argwhere(labels == 1), axis=0)
                b = np.min(np.argwhere(labels == 2), axis=0)
                fake_target = tuple(int(eyes[1][j]+b[j]-a[j]) for j in range(2))
                options = [o for o in options if o[2] == fake_target]
        if not options:
            return None
        rails, eye, outside = options[int(rng.integers(len(options)))]
        for p in rails:
            original[p] = line
        original[eye] = colors[mapping[i]]
        eyes[i] = eye
        used.update(rails)
        used.add(eye)
        reserved.add(outside)
    for i in range(count):
        if i in active:
            continue
        choices = [tuple(map(int, p)) for p in np.argwhere(labels == i)
                   if tuple(p) not in reserved and tuple(p) != (0, 0)]
        if not choices:
            return None
        eye = choices[int(rng.integers(len(choices)))]
        original[eye] = colors[mapping[i]]
        eyes[i] = eye
    output = original.copy()
    for i, eye in eyes.items():
        output[eye] = colors[mapping[mapping[i]]] if i in active else colors[i]
    anchor = int(labels[0, 0])
    order = [anchor]+[i for i in range(count) if i != anchor]
    pieces = [_tight(original, map(tuple, np.argwhere(labels == i))) for i in order]
    for p, i in zip(pieces, order):
        color = colors[i]
        if np.sum(p == color) <= np.sum(p >= 0)/2:
            return None
    return pieces, output


def _scatter(rng, pieces, bg, anchor_rotation):
    rotations = [anchor_rotation]+list(map(int, rng.integers(4, size=len(pieces)-1)))
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    full = (1 << 900)-1
    first_col = sum(1 << (r*30) for r in range(30))
    last_col = first_col << 29
    for attempt in range(12):
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt > 3:
            order = list(map(int, rng.permutation(order)))
        canvas = np.full((30, 30), bg, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        placements = []
        aspect = float(rng.uniform(.7, 1.4))
        for i in order:
            p = oriented[i]
            h, w = p.shape
            coords = [(int(r), int(c)) for r, c in np.argwhere(p >= 0)]
            base = sum(1 << (r*30+c) for r, c in coords)
            choices = []
            for r in range(31-h):
                for c in range(31-w):
                    mask = base << (r*30+c)
                    if not mask & blocked:
                        hr, wc = max(maxr, r+h), max(maxc, c+w)
                        score = hr*wc+abs(hr-aspect*wc)*2+float(rng.random())*20
                        choices.append((score, r, c, mask))
            if not choices:
                break
            _, r, c, mask = min(choices)
            horizontal = mask | ((mask & ~first_col) >> 1) | ((mask & ~last_col) << 1)
            blocked |= (horizontal | (horizontal << 30) | (horizontal >> 30)) & full
            placements.append((i, r, c))
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if len(placements) != len(pieces):
            continue
        for i, r, c in placements:
            p = oriented[i]
            view = canvas[r:r+p.shape[0], c:c+p.shape[1]]
            view[p >= 0] = p[p >= 0]
        coords = np.argwhere(canvas != bg)
        low, high = coords.min(axis=0), coords.max(axis=0)+1
        crop = canvas[low[0]:high[0], low[1]:high[1]]
        h, w = crop.shape
        H = h+int(rng.integers(min(4, 30-h)+1))
        W = w+int(rng.integers(min(4, 30-w)+1))
        largest = max(int(np.sum(crop == c)) for c in np.unique(crop) if c != bg)
        area = int(np.sum(crop != bg))
        while H*W-area <= largest:
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                break
        result = np.full((H, W), bg, dtype=np.int8)
        r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
        result[r:r+h, c:c+w] = crop
        return result
    return None


def _make_pair(rng, side, count, palette, rotation, witness=False):
    for attempt in range(180):
        if witness or attempt >= 120:
            labels = _frame_partition(rng, side, count)
        else:
            method = int(rng.integers(3))
            labels = (_frame_partition(rng, side, count) if method == 0 else
                      _bands_partition(rng, side, count) if method == 1 else
                      _growth_partition(rng, side, count))
        if labels is None:
            continue
        if any(np.sum(labels == i) < 5 or len(_components(map(tuple, np.argwhere(labels == i)), D4)) != 1
               for i in range(count)):
            continue
        annotated = _annotate(rng, labels, count, palette, witness or attempt >= 120)
        if annotated is None:
            continue
        pieces, expected = annotated
        inp = _scatter(rng, pieces, palette[0], rotation)
        if inp is None:
            continue
        try:
            facts = _read_input(inp)
            answers = _pack(facts)
        except AssertionError:
            continue
        if len(answers) != 1 or not np.array_equal(answers[0], expected):
            continue
        if witness and (len(_pack(facts, use_locks=False)) < 2
                        or _pack(facts, direct=True)
                        or len(_pack(facts, plain_body=False)) < 2):
            continue
        return {'input': inp, 'output': expected}
    raise RuntimeError("Could not construct a uniquely locked relational jigsaw")


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    varied_palette = bool(rng.random() < .65)
    scale_shift = bool(rng.random() < .65)
    held = int(rng.integers(4))
    train_sides = [int(rng.integers(9, 12)), int(rng.integers(11, 13))]
    train_sides += list(map(int, rng.integers(9, 13, size=nt-2)))
    if scale_shift:
        test_sides = list(map(int, rng.integers(13, 16, size=ne)))
    else:
        test_sides = list(map(int, rng.integers(11, 14, size=ne)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        local = index if split == 'train' else index-nt
        side = train_sides[local] if split == 'train' else test_sides[local]
        count = 3 if index == 0 else 4 if index == 1 else int(rng.choice([3, 4, 5]))
        if count == 5:
            side = max(side, 12)
        rotation = (int(rng.choice([k for k in range(4) if k != held]))
                    if split == 'train' else held)
        # At least one rotated anchor, irrespective of a randomly held-out corner.
        if index == 0:
            rotation = next(k for k in (1, 2, 3) if k != held)
        palette = list(map(int, rng.permutation(10))) if varied_palette else base.copy()
        if varied_palette and index == 1 and palette[:3] == previous_palette[:3]:
            palette = palette[1:]+palette[:1]
        previous_palette = palette
        for _ in range(8):
            pair = _make_pair(rng, side, count, palette, rotation, witness=(index == 0))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("Could not create distinct inputs")
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    return puzzle


def _features(pair):
    facts = _read_input(pair['input'])
    records, mapping = facts['records'], facts['relations']
    active = {r['body'] for r in records if r['target'] is not None}
    rail_lengths = tuple(sorted(int(np.sum(r['array'] == facts['line'])) for r in records if r['target'] is not None))
    bends = 0
    holes, silhouette_keys = 0, []
    for r in records:
        support = r['array'] >= 0
        padded = np.pad(support, 1)
        holes += len(_components(map(tuple, np.argwhere(~padded)), D4))-1
        variants = [_key(np.rot90(support.astype(np.int8), k)) for k in range(4)]
        silhouette_keys.append(min(variants))
        cells = set(map(tuple, np.argwhere(r['array'] == facts['line'])))
        for p in cells:
            adj = [(p[0]+dr, p[1]+dc) for dr, dc in D4 if (p[0]+dr, p[1]+dc) in cells]
            bends += len(adj) == 2 and (adj[0][0]+adj[1][0], adj[0][1]+adj[1][1]) != (2*p[0], 2*p[1])
    return dict(side=facts['side'], count=len(records), active=len(active),
                retained_donor=any(mapping[b] in active for b in active),
                remote_donor=any(mapping[b] not in active for b in active),
                shared_donor=len({mapping[b] for b in active}) < len(active),
                many_to_one=len(set(mapping.values())) < len(mapping),
                lengths=rail_lengths, bends=int(bends),
                holes=holes, twin_supports=len(set(silhouette_keys)) < len(silhouette_keys),
                roles=(facts['background'], facts['marker'], facts['line']),
                areas=tuple(sorted(int(np.sum(r['array'] >= 0)) for r in records)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _counts(nt, ne)
    seen, training_facts = set(), []
    ambiguous_witness = simultaneous_witness = rotation_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            facts = _read_input(inp)
            records, mapping = facts['records'], facts['relations']
            active = {r['body'] for r in records if r['target'] is not None}
            _require(2 <= len(active) < len(records), "Show selected eyes and donor-only eyes")
            _require(out.shape == (facts['side'], facts['side']), "Area-derived output size")
            _require(out[0, 0] == facts['marker'], "Marker must anchor the upper-left")
            _require(not np.any(out == facts['background']), "No discarded canvas in answer")
            _require(np.sum(out == facts['line']) == np.sum(inp == facts['line']), "Preserve every rail cell")
            _require(np.sum(out == facts['marker']) == 1, "Preserve the marker")
            # Exact color accounting for simultaneous eye updates and donor-eye
            # erasure; other piece cells are rigidly preserved by the packer.
            expected_counts = {c: int(np.sum(inp == c)) for c in range(10)}
            expected_counts[facts['background']] = 0
            for r in records:
                b = r['body']
                expected_counts[mapping[b]] -= 1
                expected_counts[mapping[mapping[b]] if b in active else b] += 1
            _require(all(np.sum(out == c) == expected_counts[c] for c in range(10)), "Wrong eye color accounting")
            answers = _pack(facts)
            _require(len(answers) == 1 and np.array_equal(answers[0], out), "Wrong or ambiguous locked packing")
            if split == 'train':
                training_facts.append(_features(pair))
                simultaneous_witness |= any(mapping[b] in active and mapping[mapping[b]] != mapping[b] for b in active)
                rotation_witness |= not bool(_pack(facts, rotations=False, limit=1))
                if not ambiguous_witness and len(_pack(facts, use_locks=False)) >= 2:
                    _require(not _pack(facts, direct=True), "Witness must distinguish one-hop locks")
                    _require(len(_pack(facts, plain_body=False)) >= 2,
                             "Witness must distinguish body identity from an eye pixel's color")
                    ambiguous_witness = True
    _require(ambiguous_witness, "Training must show geometry-only ambiguity resolved by two-hop locks")
    _require(simultaneous_witness, "Training must include a selected donor whose eye changes")
    _require(rotation_witness, "Training must require rigid quarter-turn rotation")
    _require(len({f['count'] for f in training_facts}) >= 2, "Teach variable fragment counts")


def self_test():
    """Bounded semantic, interface, and geometric diversity checks on 42 seeds."""
    puzzle_keys, rows, shapes, structures = set(), [], set(), set()
    first_key = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        pk = tuple((_key(p['input']), _key(p['output']))
                   for s in ('train', 'test') for p in puzzle[s])
        puzzle_keys.add(pk)
        if seed == 0:
            first_key = pk
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                rows.append(_features(pair))
                shapes.add(np.asarray(pair['input']).shape)
                out = np.asarray(pair['output'])
                colors = {}
                normalized = tuple(colors.setdefault(int(c), len(colors)) for c in out.flat)
                structures.add((out.shape, normalized))
                _require(np.array_equal(solve(pair['input']), out), "Input-only solve mismatch")
    _require(len(puzzle_keys) == 42, "Seeds must vary complete puzzles")
    _require(len(shapes) >= 30, "Vary scattering geometry and dimensions")
    _require(len(structures) >= 100, "Vary colored assembly geometry beyond palette labels")
    _require({3, 4, 5} <= {r['count'] for r in rows}, "Exercise fragment counts")
    _require(len({r['areas'] for r in rows}) >= 60, "Vary piece area distributions")
    _require(len({r['roles'] for r in rows}) >= 30, "Vary structural palette roles")
    _require(any(r['bends'] for r in rows), "Exercise bent ports")
    _require(any(r['many_to_one'] for r in rows), "Exercise non-permutation relations")
    _require(any(r['active'] >= 3 for r in rows), "Exercise additional locks")
    _require(any(r['holes'] == 0 for r in rows), "Go beyond enclosing-cavity geometry")
    _require(any(r['twin_supports'] for r in rows), "Exercise interchangeable supports")
    _require(any(r['remote_donor'] for r in rows), "Exercise donor-only reads")
    _require(any(r['retained_donor'] for r in rows), "Exercise simultaneous selected donors")
    first = generate_puzzle(0)
    _require(tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in first[s]) == first_key,
             "Seed reproducibility")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            # Generation is already exercised extensively above. This checks all
            # accepted count arguments without multiplying the bounded suite.
            _require(_counts(nt, ne) == (nt, ne), "Legal counts rejected")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted answer accepted")
    # Exercise the solver's sixth-fragment and side-16 extension using the same
    # visible convention, beyond the public sampler's current 3..5 / 9..15 range.
    extended = _make_pair(np.random.default_rng(4242), 16, 6, list(range(10)), 3, witness=True)
    _require(np.array_equal(solve(extended['input']), extended['output']), "Six-fragment extension failed")
    return {'seeds': 42, 'pairs': len(rows), 'input_shapes': len(shapes),
            'normalized_answers': len(structures), 'area_profiles': len({r['areas'] for r in rows}),
            'counts': sorted({r['count'] for r in rows}),
            'lock_counts': sorted({r['active'] for r in rows}),
            'bent_ports_pairs': sum(r['bends'] > 0 for r in rows),
            'non_permutation_pairs': sum(r['many_to_one'] for r in rows),
            'shared_donor_pairs': sum(r['shared_donor'] for r in rows),
            'cavity_free_pairs': sum(r['holes'] == 0 for r in rows),
            'six_fragment_extension': True}


if __name__ == '__main__':
    print(self_test())
