"""Rotation-addressed sockets: unary turns select a rigid key's destination.

Only standard library and NumPy. There is no file access or persistent state in
generation or solving. See task_spec.md for the complete visible grammar.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEAR = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    _require(_integer(a) and 2 <= a <= 10, "num_train must be an integer in 2..10")
    _require(_integer(b) and 1 <= b <= 4, "num_test must be an integer in 1..4")
    _require(a > b, "num_train must exceed num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _tight(a, mask):
    points = np.argwhere(mask)
    lo, hi = points.min(0), points.max(0) + 1
    view = a[lo[0]:hi[0], lo[1]:hi[1]]
    cut = mask[lo[0]:hi[0], lo[1]:hi[1]]
    return np.where(cut, view, -1).astype(np.int8)


def _connected(mask, neighbors=EDGE):
    points = set(map(tuple, np.argwhere(mask).tolist()))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in points:
                points.remove(p)
                stack.append(p)
    return not points


def _components(a, background):
    unseen = set(map(tuple, np.argwhere(a != background).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEAR:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    cells.append(p)
        mask = np.zeros(a.shape, dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        result.append((_tight(a, mask), tuple(cells)))
    return result


def _rotations(a):
    result, seen = [], set()
    for k in range(4):
        b = np.rot90(a, -k).copy()
        key = _key(b)
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


def _silhouette(a):
    return np.where(a >= 0, 1, -1).astype(np.int8)


def _shape_class(a):
    return min(_key(b) for b in _rotations(_silhouette(a)))


def _read_input(input_grid):
    a = _grid(input_grid)
    colors, frequencies = np.unique(a, return_counts=True)
    _require(int(np.sum(frequencies == frequencies.max())) == 1, "background must be the unique modal color")
    background = int(colors[frequencies.argmax()])
    singleton_colors = [int(c) for c, n in zip(colors, frequencies) if n == 1 and c != background]
    _require(len(singleton_colors) == 1, "one singleton marker color required")
    marker = singleton_colors[0]
    pieces, counters, locations = {}, {}, {}
    for component, cells in _components(a, background):
        palette = set(map(int, component[component >= 0]))
        # A counter is an entire detached monochrome straight component.
        if len(palette) == 1 and min(component.shape) == 1:
            color = next(iter(palette))
            _require(bool(np.all(component >= 0)) and color != marker, "invalid counter")
            _require(color not in counters and 1 <= len(cells) <= 9, "one counter of length 1..9 per carrier")
            counters[color] = len(cells)
        else:
            carriers = palette - {marker}
            _require(len(carriers) == 1, "a rigid piece has one carrier and possibly the marker")
            color = next(iter(carriers))
            _require(color not in pieces and min(component.shape) >= 2, "one two-dimensional piece per carrier")
            _require(int(np.sum(component >= 0)) >= 3, "piece area must be at least three")
            pieces[color] = component
            locations[color] = min(r for r, _ in cells), min(c for _, c in cells)
    _require(3 <= len(pieces) <= 6, "three to six pieces required")
    _require(set(counters) <= set(pieces), "counter without matching piece")
    _require(bool(counters), "at least one command required")
    anchors = [c for c, p in pieces.items() if np.any(p == marker)]
    _require(len(anchors) == 1, "marker must belong to one rigid piece")
    area = sum(int(np.sum(p >= 0)) for p in pieces.values())
    side = math.isqrt(area)
    _require(side * side == area and 1 <= side <= 30, "piece area must be a square")
    return dict(background=background, marker=marker, pieces=pieces,
                counters=counters, anchor=anchors[0], side=side, locations=locations)


def _turned(facts, counts=None):
    counts = facts['counters'] if counts is None else counts
    return {c: np.rot90(p, -counts.get(c, 0)).copy() for c, p in facts['pieces'].items()}


def _pack(pieces, side, marker, free_turns=False, limit=2):
    """Exhaust all legal translations; free_turns is an explicit ablation.

    Candidate placements are bit masks. Stop after two distinct colored results:
    that suffices to reject ambiguity, without enumerating its permutations.
    """
    colors = sorted(pieces)
    anchors = [i for i, c in enumerate(colors) if np.any(pieces[c] == marker)]
    if len(anchors) != 1:
        return []
    anchor = anchors[0]
    options = []
    for i, color in enumerate(colors):
        variants = _rotations(pieces[color]) if free_turns else [pieces[color]]
        placements = {}
        for p in variants:
            h, w = p.shape
            if h > side or w > side:
                continue
            if i == anchor and p[0, 0] != marker:
                continue
            cells = tuple((int(r) * side + int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0))
            base = sum(1 << pos for pos, _ in cells)
            offsets = [0] if i == anchor else [r * side + c for r in range(side - h + 1) for c in range(side - w + 1)]
            for offset in offsets:
                mask = base << offset
                placements.setdefault(mask, (mask, offset, cells))
        if not placements:
            return []
        options.append(list(placements.values()))
    full = (1 << (side * side)) - 1
    placed = [None] * len(colors)
    answers, signatures = [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            signature = tuple(p[0] for p in placed)
            if signature not in signatures:
                signatures.add(signature)
                out = np.empty((side, side), dtype=np.int8)
                for _, offset, cells in placed:
                    for pos, value in cells:
                        out.flat[pos + offset] = value
                answers.append(out)
            return
        blank = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                if candidate[0] & blank and not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    for candidate in options[anchor]:
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(colors)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: read unary commands, turn rigid pieces, then translate to tile.

    No extra turns, reflection, scaling, recoloring, or overlap is permitted.
    """
    f = _read_input(input_grid)
    answers = _pack(_turned(f), f['side'], f['marker'])
    _require(len(answers) == 1, "commands must select exactly one anchored colored tiling")
    return answers[0]


def _random_key(rng, allow_diagonal=True):
    """Grow a fresh small key, including occasional corner-only attachments."""
    for _ in range(80):
        h, w = map(int, rng.integers(2, 5, size=2))
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        wanted = int(rng.integers(3, min(h * w, 11) + 1))
        diagonal = bool(allow_diagonal and rng.random() < .18)
        for _ in range(wanted - 1):
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in (NEAR if diagonal else EDGE):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            choices = sorted(frontier)
            mask[choices[int(rng.integers(len(choices)))]] = True
        shape = _tight(np.ones(mask.shape, dtype=np.int8), mask)
        if min(shape.shape) >= 2 and len(_rotations(shape)) == 4:
            return shape
    # A connected, chiral L with four distinct poses is a bounded safety net.
    return np.array([[1, -1], [1, -1], [1, 1]], dtype=np.int8)


def _blueprint(rng, side, marker, colors, teaching=False, keycount=None,
               body_choices=(1, 2), allow_diagonal=True):
    """Build a square with repeated rigid keys and one/two enclosing scaffolds.

    This is a procedural socket construction, not a call to either parent.
    The two teaching sockets have opposite poses, enabling a command swap.
    """
    for _ in range(160):
        bodies = int(rng.choice(body_choices))
        count = (int(rng.integers(3 if teaching else 2, min(4, len(colors) - bodies) + 1))
                 if keycount is None else int(keycount))
        carriers = colors[:bodies + count]
        key = _random_key(rng, allow_diagonal)
        first = int(rng.integers(4))
        poses = [first, (first + 2) % 4]
        poses += [q for q in rng.permutation(4) if q not in poses][:count - 2]
        holes = []
        blocked = np.zeros((side, side), dtype=bool)
        failed = False
        for q in poses:
            p = np.rot90(key, -q)
            h, w = p.shape
            choices = []
            # Edge sockets and interior sockets are both allowed. The marker
            # cell and a clear one-cell moat between keys are protected.
            for r in range(side - h + 1):
                for c in range(side - w + 1):
                    cells = [(r + int(y), c + int(x)) for y, x in np.argwhere(p >= 0)]
                    if (0, 0) in cells or any(blocked[y, x] for y, x in cells):
                        continue
                    choices.append(cells)
            if not choices:
                failed = True
                break
            cells = choices[int(rng.integers(len(choices)))]
            mask = np.zeros((side, side), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
                blocked[max(0, r-1):min(side, r+2), max(0, c-1):min(side, c+2)] = True
            holes.append(mask)
        if failed:
            continue
        remaining = ~np.logical_or.reduce(holes)
        if not _connected(remaining):
            continue
        labels = np.zeros((side, side), dtype=np.int8)
        if bodies == 2:
            # A stepped seam splits the scaffold into two genuine rigid bodies.
            cut = int(rng.integers(2, side - 1))
            seam = []
            for _ in range(side):
                cut = int(np.clip(cut + rng.choice([-1, 0, 0, 1]), 2, side-2))
                seam.append(cut)
            for r, c in enumerate(seam):
                labels[r, c:] = 1
            if rng.random() < .5:
                labels = labels.T.copy()
        for i, mask in enumerate(holes, bodies):
            labels[mask] = i
        masks = [labels == i for i in range(bodies + count)]
        if any(int(m.sum()) < 3 or not _connected(m, NEAR if allow_diagonal else EDGE) for m in masks):
            continue
        out = np.empty((side, side), dtype=np.int8)
        for color, mask in zip(carriers, masks):
            out[mask] = color
        out[0, 0] = marker
        pieces = {c: _tight(out, m) for c, m in zip(carriers, masks)}
        if any(min(p.shape) < 2 for p in pieces.values()):
            continue
        key_class = _shape_class(pieces[carriers[bodies]])
        if any(_shape_class(pieces[c]) == key_class for c in carriers[:bodies]):
            continue
        answers = _pack(pieces, side, marker)
        if len(answers) != 1 or not np.array_equal(answers[0], out):
            continue
        return dict(output=out, pieces=pieces, twins=tuple(carriers[bodies:bodies+2]),
                    bodies=bodies, poses=poses)
    raise RuntimeError("bounded socket construction exhausted")


def _scatter(rng, pieces, counters, background, forced_orientation=None):
    """Pack real masks, with an eight-neighbor gap; never save search traces."""
    forced_orientation = forced_orientation or {}
    components = []
    for c, p in pieces.items():
        components.append((p, False))
    for c, k in sorted(counters.items(), key=lambda item: -item[1]):
        vertical = forced_orientation.get(c, bool(rng.integers(2)))
        components.append((np.full((k, 1) if vertical else (1, k), c, dtype=np.int8), True))
    components.sort(key=lambda item: (-item[0].size, item[1]))
    width = 30
    full = (1 << 900) - 1
    left = sum(1 << (30 * r) for r in range(30))
    right = left << 29
    for attempt in range(10):
        canvas = np.full((30, 30), background, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        okay = True
        spread = float(rng.uniform(8, 180))
        for index, (p, _) in enumerate(components):
            h, w = p.shape
            coords = tuple((int(r), int(c)) for r, c in np.argwhere(p >= 0))
            base = sum(1 << (r * width + c) for r, c in coords)
            choices = []
            for r in range(31 - h):
                for c in range(31 - w):
                    mask = base << (r * width + c)
                    if mask & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    score = hr * wc + abs(hr - wc) * float(rng.uniform(.2, 1.5)) + float(rng.random()) * spread
                    choices.append((score, r, c, mask))
            if not choices:
                okay = False
                break
            _, r, c, mask = min(choices)
            view = canvas[r:r+h, c:c+w]
            view[p >= 0] = p[p >= 0]
            horizontal = mask | ((mask & ~left) >> 1) | ((mask & ~right) << 1)
            blocked |= (horizontal | (horizontal << width) | (horizontal >> width)) & full
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not okay:
            continue
        crop = canvas[:maxr, :maxc]
        H = min(30, maxr + int(rng.integers(5)))
        W = min(30, maxc + int(rng.integers(5)))
        foreground = int(np.sum(crop != background))
        largest = max(Counter(map(int, crop[crop != background])).values())
        while H * W - foreground <= largest:
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                break
        if H * W - foreground <= largest:
            continue
        top = int(rng.integers(H - maxr + 1))
        col = int(rng.integers(W - maxc + 1))
        result = np.full((H, W), background, dtype=np.int8)
        result[top:top+maxr, col:col+maxc] = crop
        return result
    raise RuntimeError("bounded scatter exhausted")


def _make_pair(rng, blueprint, marker, background, counters, swap=False, forced=None):
    out = blueprint['output'].copy()
    pieces = {c: p.copy() for c, p in blueprint['pieces'].items()}
    if swap:
        a, b = blueprint['twins']
        ma, mb = out == a, out == b
        out[ma], out[mb] = b, a
        pieces[a], pieces[b] = pieces[b].copy(), pieces[a].copy()
        pieces[a][pieces[a] >= 0] = a
        pieces[b][pieces[b] >= 0] = b
    # Generate the scattered pose by inverting the visible clockwise command.
    initial = {c: np.rot90(p, counters.get(c, 0)).copy() for c, p in pieces.items()}
    inp = _scatter(rng, initial, counters, background, forced)
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    marker, background = palette[:2]
    colors = palette[2:8]
    side = int(rng.integers(7, 11))
    teaching = _blueprint(rng, side, marker, colors, teaching=True)
    x, y = teaching['twins']
    anchor = colors[0]
    extra = [c for c in teaching['pieces'] if c not in (x, y, anchor)]
    # Twin commands exchange opposite sockets. Raw 5 and 7 demonstrate cycling,
    # rather than assuming every long bar is a new kind of instruction.
    first = {x: 1, y: 3, anchor: 2}
    second = {x: 7, y: 5, anchor: 4}
    for c in extra:
        first[c] = second[c] = 0
    first = {c: n for c, n in first.items() if n}
    second = {c: n for c, n in second.items() if n}
    p0 = _make_pair(rng, teaching, marker, background, first, forced={y: False})
    background2 = next(c for c in palette if c not in set(teaching['pieces']) | {marker, background})
    p1 = _make_pair(rng, teaching, marker, background2, second, swap=True, forced={x: True})
    puzzle = {'train': [p0, p1], 'test': []}
    used = {_key(p0['input']), _key(p1['input'])}
    # The test shift holds out a repeated-key count. Additional training
    # examples retain the teaching count; tests use a different count.
    teaching_keys = len(teaching['pieces']) - teaching['bodies']
    two_bodies_seen = teaching['bodies'] == 2
    diagonal_seen = any(not _connected(p >= 0) for p in teaching['pieces'].values())
    for split, count in (('train', a - 2), ('test', b)):
        for _ in range(count):
            for retry in range(24):
                bg = int(rng.choice([c for c in range(10) if c != marker]))
                pool = list(map(int, rng.permutation([c for c in range(10) if c not in (marker, bg)])))[:6]
                n = int(rng.integers(7, 13 if split == 'train' else 15))
                nkeys = (teaching_keys if split == 'train' else
                         int(rng.choice([k for k in (2, 3, 4) if k != teaching_keys])))
                body_choices = (1, 2) if split == 'train' or two_bodies_seen else (1,)
                bp = _blueprint(rng, n, marker, pool, keycount=nkeys, body_choices=body_choices,
                                allow_diagonal=split == 'train' or diagonal_seen)
                commands = {c: int(rng.integers(0, 10)) for c in bp['pieces']}
                # Both splits can contain long counts, absent commands, and
                # full cycles. A nonzero twin turn ensures useful control.
                commands[bp['twins'][0]] = int(rng.choice([1, 2, 3, 5, 6, 7, 9]))
                commands = {c: k for c, k in commands.items() if k}
                pair = _make_pair(rng, bp, marker, bg, commands)
                if _key(pair['input']) not in used:
                    used.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    if split == 'train':
                        two_bodies_seen |= bp['bodies'] == 2
                        diagonal_seen |= any(not _connected(p >= 0) for p in bp['pieces'].values())
                    break
            else:
                raise RuntimeError("bounded distinct-pair schedule exhausted")
    return puzzle


def _coordinate_turn(piece, k):
    """Validation uses cell coordinates instead of generation's array rotation."""
    cells = [(int(r), int(c), int(piece[r, c])) for r, c in np.argwhere(piece >= 0)]
    h, w = piece.shape
    for _ in range(k % 4):
        cells = [(c, h - 1 - r, value) for r, c, value in cells]
        h, w = w, h
    result = np.full((h, w), -1, dtype=np.int8)
    for r, c, value in cells:
        result[r, c] = value
    return result


def _features(pair, check=True):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    f = _read_input(inp)
    marker, n = f['marker'], f['side']
    if check:
        _require(out.shape == (n, n) and out[0, 0] == marker, "wrong output extent or anchor")
        _require(not np.any(out == f['background']), "output must be completely tiled")
        for c, piece in f['pieces'].items():
            omask = (out == c) | ((out == marker) if c == f['anchor'] else False)
            _require(int(omask.sum()) == int(np.sum(piece >= 0)), "rigid area changed")
            expected = _coordinate_turn(piece, f['counters'].get(c, 0))
            _require(np.array_equal(_tight(out, omask), expected), "piece turned incorrectly, reflected, or deformed")
        answers = _pack(_turned(f), n, marker)
        _require(len(answers) == 1 and np.array_equal(answers[0], out), "not the unique command-constrained assembly")
        # This is a semantic ablation: B with free rotations cannot determine
        # the colors of congruent keys. A genuinely removes that ambiguity.
        free = _pack(f['pieces'], n, marker, free_turns=True)
        _require(len(free) == 2, "omitting counters must leave multiple colored answers")
        # Ablate translation/packing: turn at each original component's origin
        # and merely crop. Collisions or a non-square/gappy result reject it.
        in_place, collision = {}, False
        for c, p in _turned(f).items():
            top, left = f['locations'][c]
            for r, col in np.argwhere(p >= 0):
                position = top + int(r), left + int(col)
                collision |= position in in_place
                in_place[position] = int(p[r, col])
        top = min(r for r, _ in in_place)
        left = min(c for _, c in in_place)
        expected_cells = {(r, c): int(out[r, c]) for r in range(n) for c in range(n)}
        cropped_cells = {(r-top, c-left): value for (r, c), value in in_place.items()}
        _require(collision or cropped_cells != expected_cells, "omitting assembly must change the answer")
    classes = Counter(_shape_class(p) for p in f['pieces'].values())
    repeated = max(classes.values())
    _require(repeated >= 2, "at least two rotationally congruent pieces required")
    absent = len(f['pieces']) - len(f['counters'])
    edge = interior = 0
    for c in f['pieces']:
        if c == f['anchor']:
            continue
        mask = out == c
        touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        edge += touches
        interior += not touches
    f.update(repeated=repeated, absent=absent, edge_pieces=edge, interior_pieces=interior,
             diagonal=any(not _connected(p >= 0) for p in f['pieces'].values()))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "train/test schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            key = _key(_grid(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts[split].append(_features(pair))
    _require(len({f['marker'] for s in facts.values() for f in s}) == 1, "one marker convention per puzzle")
    tr = facts['train']
    _require({1, 2, 3, 4, 5, 7} <= {k for f in tr for k in f['counters'].values()}, "training must expose direction and cycling")
    # Verify the paired intervention from arrays, without trusting a latent flag.
    out0, out1 = (np.asarray(p['output']) for p in puzzle['train'][:2])
    _require(out0.shape == out1.shape and not np.array_equal(out0, out1), "demonstrate command-selected color routing")
    changed = set(map(int, out0[out0 != out1]))
    _require(len(changed) == 2, "teaching intervention exchanges exactly two carrier colors")
    c, d = sorted(changed)
    _require(np.array_equal(out0 == c, out1 == d) and np.array_equal(out0 == d, out1 == c), "teaching socket swap")
    _require(np.array_equal(tr[0]['pieces'][c], tr[1]['pieces'][c]) and np.array_equal(tr[0]['pieces'][d], tr[1]['pieces'][d]), "the command intervention holds the initial key poses fixed")
    _require(len({f['repeated'] for f in tr}) == 1, "training key-count regime")
    _require(all(f['repeated'] != tr[0]['repeated'] for f in facts['test']), "test must change the repeated-key count")
    _require(any(f['absent'] for f in tr), "training must show the unmarked default")
    _require(not any(f['diagonal'] for f in facts['test']) or any(f['diagonal'] for f in tr), "show diagonal rigidity before testing it")
    # In this constructor, repeated keys are one congruence class; other pieces
    # are the scaffold. This is a visible shape statistic, not saved metadata.
    _require(not any(len(f['pieces']) - f['repeated'] == 2 for f in facts['test']) or
             any(len(f['pieces']) - f['repeated'] == 2 for f in tr), "show split scaffold before testing it")


def self_test():
    """42 seeds, bounded scalar/hash diagnostics, plus mutations and counts.

    No artifacts are written, and no collection of puzzle payloads is retained.
    """
    signatures, shapes, normalized = set(), set(), set()
    backgrounds, counts, repeated, sides, lengths = set(), set(), set(), set(), set()
    edge = interior = diagonal = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair, check=False)
                backgrounds.add(f['background'])
                counts.add(len(f['pieces']))
                repeated.add(f['repeated'])
                sides.add(f['side'])
                lengths.update(f['counters'].values())
                edge += f['edge_pieces'] > 0
                interior += f['interior_pieces'] > 0
                diagonal += f['diagonal']
                _require(np.array_equal(solve(pair['input']), pair['output']), "solver mismatch")
                for role in ('input', 'output'):
                    g = pair[role]
                    digest.update(bytes(g.shape) + g.tobytes())
                    shapes.add((role, g.shape))
                mapping = {}
                pattern = bytes(mapping.setdefault(int(v), len(mapping)) for v in pair['output'].flat)
                normalized.add((pair['output'].shape, hashlib.sha256(pattern).digest()))
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(normalized) >= 100, "structural diversity")
    _require(len(shapes) >= 35 and sides >= set(range(7, 15)), "geometry coverage")
    _require(backgrounds == set(range(10)) and counts == {3, 4, 5, 6}, "palette/count coverage")
    _require(repeated == {2, 3, 4} and lengths == set(range(1, 10)), "command/key coverage")
    _require(edge >= 20 and interior >= 20 and diagonal >= 5, "socket/connectivity coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(123, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, "counts not honored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(7, 2, 1), generate_puzzle(7, 2, 1)
    _require(all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "nondeterminism")
    p['test'][0]['output'][0, 0] = int(p['train'][0]['output'][1, 1])
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
