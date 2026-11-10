"""Repairable jigsaw matter: joint shape correction and anchored exact cover.

Only Python's standard library and NumPy are used. No files, seeds, examples,
parent code, or generation metadata are consulted by solve().
"""
from itertools import combinations
from math import isqrt
import hashlib
import json

import numpy as np

_EDGE = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'Invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid must contain integers')
    _require(bool(((a >= 0) & (a <= 9)).all()), 'Invalid color')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _EIGHT:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
                    found.append(p)
        result.append(found)
    return result


def _connected(mask):
    return bool(mask.any()) and len(_components(mask)) == 1


def _tight(a):
    rr, cc = np.nonzero(a >= 0)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    result = np.zeros_like(mask)
    for m, out in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            out[:, k:k+triples.shape[1]] |= triples
    return result


def _cues(piece):
    """Visible possible corrections; a cue is evidence, not a forced edit.

    Return (three empty cells, two weak spur cells, two retained endpoints).
    Everything is measured on the unedited piece. A genuine diagonal attachment
    can satisfy this exact same test and still have to remain unchanged.
    """
    mask = piece >= 0
    strong = _strong(mask)
    found, seen = [], set()
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
                continue
            anchors = ((r, c), (r+4*dr, c+4*dc))
            if piece[anchors[0]] != piece[anchors[1]]:
                continue
            carrier = int(piece[anchors[0]])
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr + side*dc, toward*dc + side*dr
                    spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                    if not all(_at(mask, rr, cc) and not strong[rr, cc]
                               and int(piece[rr, cc]) == carrier for rr, cc in spur):
                        continue
                    if _at(mask, ar+3*sr, ac+3*sc):
                        continue
                    sig = (gap, tuple(sorted(spur)))
                    if sig not in seen:
                        seen.add(sig)
                        found.append((gap, spur, anchors))
    return found


def _variants(piece, budget):
    cues = _cues(piece)
    variants, seen = [], set()
    for count in range(min(budget, len(cues))+1):
        for indices in combinations(range(len(cues)), count):
            used, valid = set(), True
            for i in indices:
                footprint = set(sum(cues[i], ()))
                if used & footprint:
                    valid = False
                    break
                used.update(footprint)
            if not valid:
                continue
            edited = piece.copy()
            for i in indices:
                gap, spur, anchors = cues[i]
                color = int(piece[anchors[0]])
                for p in gap:
                    edited[p] = color
                for p in spur:
                    edited[p] = -1
            if not _connected(edited >= 0):
                continue
            edited = _tight(edited)
            key = _key(edited)
            if key not in seen:
                seen.add(key)
                variants.append((edited, count))
    return variants


def _read_input(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(int(np.sum(counts == counts.max())) == 1, 'Background must be uniquely dominant')
    bg = int(values[np.argmax(counts)])
    singles = [int(c) for c, n in zip(values, counts) if n == 1 and c != bg]
    _require(len(singles) == 1, 'Need exactly one singleton marker')
    marker = singles[0]
    pieces, carriers = [], []
    for cells in _components(grid != bg):
        rr, cc = np.array(cells).T
        r0, c0 = int(rr.min()), int(cc.min())
        piece = np.full((int(rr.max())-r0+1, int(cc.max())-c0+1), -1, np.int8)
        piece[rr-r0, cc-c0] = grid[rr, cc]
        colors = set(map(int, piece[piece >= 0])) - {marker}
        _require(len(colors) == 1, 'Every shard has one carrier and optional marker')
        carrier = next(iter(colors))
        _require(int(np.sum(piece == carrier)) >= 3, 'Too little carrier material')
        pieces.append(piece)
        carriers.append(carrier)
    _require(3 <= len(pieces) <= 5 and len(set(carriers)) == len(pieces), 'Need 3..5 distinct shards')
    area = int(np.sum(grid != bg))
    side = isqrt(area)
    if side*side < area:
        side += 1
    budget = side*side-area
    _require(1 <= budget <= 3 and side <= 30, 'Need one to three net-cell corrections')
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    _require(len(marked) == 1, 'Marker belongs to exactly one shard')
    return dict(grid=grid, bg=bg, marker=marker, pieces=pieces, carriers=carriers,
                side=side, budget=budget, anchor=marked[0])


def _assemblies(facts, limit=2):
    """Joint search over shape corrections and rigid placements, input only."""
    side, required = facts['side'], facts['budget']
    pieces, anchor = facts['pieces'], facts['anchor']
    all_options, indexes, cost_bounds = [], [], []
    for i, piece in enumerate(pieces):
        options, seen = [], set()
        for variant, cost in _variants(piece, required):
            for turn in range(4):
                p = np.rot90(variant, turn)
                h, w = p.shape
                if h > side or w > side:
                    continue
                if i == anchor:
                    if int(p[0, 0]) != facts['marker']:
                        continue
                    origins = [(0, 0)]
                else:
                    origins = ((r, c) for r in range(side-h+1) for c in range(side-w+1))
                base = sum(1 << (int(r)*side+int(c)) for r, c in np.argwhere(p >= 0))
                for r, c in origins:
                    mask = base << (r*side+c)
                    key = mask, cost
                    if key not in seen:
                        seen.add(key)
                        options.append(key)
        if not options:
            return []
        by_cell = [[] for _ in range(side*side)]
        for option in options:
            bits = option[0]
            while bits:
                bit = bits & -bits
                by_cell[bit.bit_length()-1].append(option)
                bits ^= bit
        all_options.append(options)
        indexes.append(by_cell)
        cost_bounds.append((min(c for _, c in options), max(c for _, c in options)))
    full = (1 << (side*side))-1
    placed = [None]*len(pieces)
    answers, signatures = [], set()

    def search(occupied, remaining, spent):
        if len(answers) >= limit or spent > required:
            return
        if spent + sum(cost_bounds[i][0] for i in remaining) > required:
            return
        if spent + sum(cost_bounds[i][1] for i in remaining) < required:
            return
        if not remaining:
            if occupied != full or spent != required:
                return
            answer = np.empty((side, side), np.int8)
            for i, option in enumerate(placed):
                bits = option[0]
                while bits:
                    bit = bits & -bits
                    answer.flat[bit.bit_length()-1] = facts['carriers'][i]
                    bits ^= bit
            answer[0, 0] = facts['marker']
            sig = answer.tobytes()
            if sig not in signatures:
                signatures.add(sig)
                answers.append(answer)
            return
        bit = (full ^ occupied) & -(full ^ occupied)
        cell = bit.bit_length()-1
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for option in indexes[i][cell]:
                mask, cost = option
                if cost+spent <= required and not mask & occupied:
                    placed[i] = option
                    search(occupied | mask, rest, spent+cost)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for option in all_options[anchor]:
        placed[anchor] = option
        search(option[0], rest, option[1])
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    answers = _assemblies(facts)
    _require(len(answers) == 1, 'Input must have exactly one corrected colored square')
    return answers[0].copy()


def _cuts(rng, side, count):
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        choices = []
        for i, (r, c, h, w) in enumerate(rectangles):
            choices += [(i, 0, k) for k in range(2, h-1) if min(k, h-k)*w >= 10]
            choices += [(i, 1, k) for k in range(2, w-1) if min(k, w-k)*h >= 10]
        if not choices:
            return None
        i, axis, k = choices[int(rng.integers(len(choices)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    # Uneven steps and notches, while retaining some long straight interfaces.
    for _ in range(int(rng.integers(side//2, side*2))):
        r, c = map(int, rng.integers(side, size=2))
        if (r, c) == (0, 0):
            continue
        dr, dc = _EDGE[int(rng.integers(4))]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < side and 0 <= nc < side):
            continue
        a, b = int(labels[r, c]), int(labels[nr, nc])
        if a == b or np.sum(labels == a) <= 8:
            continue
        labels[r, c] = b
        if not _connected(labels == a):
            labels[r, c] = a
    return labels


def _inserts(rng, side, count):
    labels = np.zeros((side, side), np.int8)
    for label in range(1, count):
        for _ in range(60):
            h = int(rng.integers(2, max(4, side//2+1)))
            w = int(rng.integers(3, max(5, side-2)))
            if rng.integers(2):
                h, w = w, h
            r = int(rng.integers(side-h+1))
            c = int(rng.integers(side-w+1))
            if (r == 0 and c == 0) or np.any(labels[r:r+h, c:c+w]):
                continue
            proposal = labels.copy()
            proposal[r:r+h, c:c+w] = label
            # Some inserts open to the boundary, others are enclosed.
            if _connected(proposal == 0):
                labels = proposal
                break
        else:
            return None
    return labels


def _damage_options(labels):
    """Local reversible edits on a clean partition, not parent generators."""
    side = labels.shape[0]
    options = []
    for owner in sorted(map(int, np.unique(labels))):
        own = labels == owner
        for r, c in np.argwhere(own):
            r, c = int(r), int(c)
            for dr, dc in ((0, 1), (1, 0)):
                if not all(_at(own, r+k*dr, c+k*dc) for k in range(5)):
                    continue
                if not (_at(own, r-dr, c-dc) or _at(own, r+5*dr, c+5*dc)):
                    continue
                gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
                for endpoint in (0, 4):
                    ar, ac = r+endpoint*dr, c+endpoint*dc
                    toward = 1 if endpoint == 0 else -1
                    for bend in (-1, 1):
                        sr, sc = toward*dr+bend*dc, toward*dc+bend*dr
                        spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                        if not all(0 <= rr < side and 0 <= cc < side and not own[rr, cc]
                                   for rr, cc in spur):
                            continue
                        if (0, 0) in gap+spur+((r, c), (r+4*dr, c+4*dc)):
                            continue
                        trial = np.where(own, owner+1, -1).astype(np.int8)
                        for p in gap:
                            trial[p] = -1
                        for p in spur:
                            trial[p] = owner+1
                        if not _connected(trial >= 0):
                            continue
                        if not any(set(g) == set(gap) and set(s) == set(spur) for g, s, _ in _cues(trial)):
                            continue
                        options.append((owner, gap, spur, ((r, c), (r+4*dr, c+4*dc))))
    return options


def _false_cue(rng, labels):
    options = _damage_options(labels)
    for index in rng.permutation(len(options)):
        owner, gap, spur, _ = options[int(index)]
        recipients = {int(labels[p]) for p in spur}
        if len(recipients) != 1:
            continue
        other = recipients.pop()
        trial = labels.copy()
        for p in gap:
            trial[p] = other
        for p in spur:
            trial[p] = owner
        if all(np.sum(trial == i) >= 6 and _connected(trial == i) for i in np.unique(labels)):
            return trial, set(gap+spur)
    return None


def _colored_pieces(labels, colors, marker):
    output = np.array(colors, np.int8)[labels]
    output[0, 0] = marker
    pieces = []
    for i in range(len(colors)):
        piece = np.full(labels.shape, -1, np.int8)
        piece[labels == i] = output[labels == i]
        pieces.append(piece)
    return output, pieces


def _scatter(rng, pieces, background):
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    # Guarantee an asymmetric rotation demonstration while allowing the other
    # pieces, including the marked one, to start in any quarter-turn pose.
    for i, p in enumerate(pieces):
        tight = _tight(p)
        if len({_key(np.rot90(tight, k)) for k in range(4)}) == 4:
            rotations[i] = int(rng.integers(1, 4))
            break
    oriented = [np.rot90(_tight(p), k).copy() for p, k in zip(pieces, rotations)]
    # Mask placement allows staggered and interleaving bounding boxes.
    full = (1 << 900)-1
    first = sum(1 << (30*r) for r in range(30))
    last = first << 29
    for attempt in range(12):
        canvas = np.full((30, 30), background, np.int8)
        blocked, maxr, maxc = 0, 0, 0
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size + float(rng.random())*8)
        okay = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            base = sum(1 << (int(r)*30+int(c)) for r, c in np.argwhere(p >= 0))
            choices = []
            for r in range(31-h):
                for c in range(31-w):
                    mask = base << (30*r+c)
                    if not mask & blocked:
                        hr, wc = max(maxr, r+h), max(maxc, c+w)
                        score = hr*wc + abs(hr-wc)*float(rng.uniform(0, 3)) + float(rng.uniform(0, 30))
                        choices.append((score, r, c, mask))
            if not choices:
                okay = False
                break
            _, r, c, mask = min(choices)
            view = canvas[r:r+h, c:c+w]
            view[p >= 0] = p[p >= 0]
            horizontal = mask | ((mask & ~first) >> 1) | ((mask & ~last) << 1)
            blocked |= (horizontal | (horizontal << 30) | (horizontal >> 30)) & full
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not okay:
            continue
        crop = canvas[:maxr, :maxc]
        H = maxr + int(rng.integers(min(4, 30-maxr)+1))
        W = maxc + int(rng.integers(min(4, 30-maxc)+1))
        material = int(np.sum(crop != background))
        largest = max(int(np.sum(crop == col)) for col in np.unique(crop) if col != background)
        while H*W-material <= largest and (H < 30 or W < 30):
            if H < 30 and (H <= W or W == 30):
                H += 1
            else:
                W += 1
        if H*W-material <= largest:
            continue
        out = np.full((H, W), background, np.int8)
        r, c = int(rng.integers(H-maxr+1)), int(rng.integers(W-maxc+1))
        out[r:r+maxr, c:c+maxc] = crop
        return out
    return None


def _make_pair(rng, side, count, repairs, palette):
    background, marker = palette[:2]
    colors = palette[2:2+count]
    for attempt in range(240):
        labels = _inserts(rng, side, count) if attempt % 3 == 0 else _cuts(rng, side, count)
        if labels is None:
            continue
        false = _false_cue(rng, labels)
        if false is None:
            continue
        labels, protected = false
        output, pieces = _colored_pieces(labels, colors, marker)
        choices = _damage_options(labels)
        used, selected = set(protected), []
        for index in rng.permutation(len(choices)):
            owner, gap, spur, anchors = choices[int(index)]
            footprint = set(gap+spur+anchors)
            if footprint & used:
                continue
            # Protect the visible false cue's immediate neighborhood too.
            if any(abs(r-pr) <= 1 and abs(c-pc) <= 1 for r, c in footprint for pr, pc in protected):
                continue
            edited = pieces[owner].copy()
            for p in gap:
                edited[p] = -1
            for p in spur:
                edited[p] = colors[owner]
            if not _connected(edited >= 0):
                continue
            pieces[owner] = edited
            used.update(footprint)
            selected.append((owner, gap, spur))
            if len(selected) == repairs:
                break
        if len(selected) != repairs:
            continue
        # Every intended corruption must still have its complete raw-input cue.
        if not all(any(set(g) == set(gap) and set(s) == set(spur)
                       for g, s, _ in _cues(pieces[owner])) for owner, gap, spur in selected):
            continue
        if sum(len(_cues(p)) for p in pieces) <= repairs:
            continue
        if not all(_connected(p >= 0) for p in pieces):
            continue
        # Check uniqueness before paying for spatial scattering.
        facts = dict(side=side, budget=repairs, pieces=[_tight(p) for p in pieces],
                     marker=marker, carriers=colors, anchor=int(labels[0, 0]))
        answers = _assemblies(facts)
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        inp = _scatter(rng, pieces, background)
        if inp is not None:
            return {'input': inp, 'output': output}
    raise RuntimeError('Bounded construction exhausted; no ambiguous fallback is emitted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    t = _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = list(map(int, rng.permutation(10)))
    # Marker stays fixed; the other roles either stay stable or change visibly.
    varied = bool(rng.integers(2))
    shift = int(rng.integers(2))
    train_count = int(rng.choice([3, 4, 5]))
    first_side = int(rng.integers(10, 14))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', n), ('test', t)):
        for j in range(total):
            if split == 'train':
                side = (first_side if j == 0 else
                        int(rng.choice([s for s in range(10, 14) if s != first_side])) if j == 1 else
                        int(rng.integers(10, 14)))
                repairs = 1 if j == 0 else 2 if j == 1 else int(rng.integers(1, 3))
                count = (train_count if j == 0 else
                         int(rng.choice([c for c in (3, 4, 5) if c != train_count])) if j == 1 else
                         int(rng.choice([3, 4, 5])))
            else:
                side = int(rng.integers(14, 16)) if shift == 0 else int(rng.integers(10, 14))
                repairs = 2 if shift == 0 else 3
                count = int(rng.choice([3, 4, 5]))
            if varied:
                pool = [c for c in range(10) if c != base_palette[1]]
                rest = list(map(int, rng.permutation(pool)))
                palette = [rest[0], base_palette[1]] + rest[1:]
            else:
                palette = base_palette
            for _ in range(8):
                pair = _make_pair(rng, side, count, repairs, palette)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not generate a distinct input')
    return puzzle


def _measure_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
    facts = _read_input(pair['input'])
    output = _grid(pair['output'])
    side = facts['side']
    _require(output.shape == (side, side), 'Wrong square size')
    _require(int(output[0, 0]) == facts['marker'] and np.sum(output == facts['marker']) == 1, 'Wrong anchor')
    _require(set(map(int, np.unique(output))) == set(facts['carriers']) | {facts['marker']}, 'Wrong output palette')
    differences = []
    for carrier in facts['carriers']:
        delta = int(np.sum(output == carrier)-np.sum(facts['grid'] == carrier))
        _require(0 <= delta <= facts['budget'], 'Carrier material can only gain one cell per repair')
        differences.append(delta)
    _require(sum(differences) == facts['budget'], 'Net material count mismatch')
    cues = sum(len(_cues(p)) for p in facts['pieces'])
    _require(cues > facts['budget'], 'Need competing repair cues')
    retained = 0
    anchor_carrier = facts['carriers'][facts['anchor']]
    for carrier in facts['carriers']:
        mask = (output == carrier)
        if carrier == anchor_carrier:
            mask |= output == facts['marker']
        _require(_connected(mask), 'Corrected shard must remain corner-connected')
        piece = np.where(mask, output, -1).astype(np.int8)
        retained += len(_cues(_tight(piece)))
    _require(retained >= 1, 'A genuine diagonal-and-bite cue must survive')
    answers = _assemblies(facts)
    _require(len(answers) == 1 and np.array_equal(answers[0], output), 'Wrong or ambiguous corrected assembly')
    # A blanket repair would add too much material even before packing.
    _require(cues != facts['budget'], 'Ignoring rigid attachments must fail')
    return dict(side=side, count=len(facts['pieces']), repairs=facts['budget'], cues=cues, retained_cues=retained,
                shape=facts['grid'].shape, palette=(facts['bg'], facts['marker'], *sorted(facts['carriers'])),
                output_structure=_normalized(output), area_ratio=max(np.sum(output == c) for c in facts['carriers']) /
                min(np.sum(output == c) for c in facts['carriers']))


def _normalized(grid):
    roles = {}
    return grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    n, t = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, 'Illegal counts')
    seen, markers, infos = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _measure_pair(pair)
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, 'Duplicate input')
            seen.add(key)
            markers.add(info['palette'][1])
            infos[split].append(info)
    _require(len(markers) == 1, 'Marker convention changes within puzzle')
    _require({1, 2} <= {x['repairs'] for x in infos['train']}, 'Training must show one and multiple repairs')
    _require(len({x['side'] for x in infos['train']}) >= 2, 'Demonstrate area-derived size')
    _require(len({x['count'] for x in infos['train']}) >= 2, 'Demonstrate variable shard count')
    _require(all(x['cues'] > x['repairs'] for x in infos['train']), 'Training must preserve genuine cues')


def _digest_puzzle(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streaming measurements; no stored runs, traces or payload logs."""
    signatures, structures, shapes, palettes = set(), set(), set(), set()
    counts, sizes, repair_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest_puzzle(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _measure_pair(pair)
                structures.add(f['output_structure'])
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                counts.add(f['count'])
                sizes.add(f['side'])
                repair_counts.add(f['repairs'])
                _require(np.array_equal(solve(pair['input']), pair['output']), 'Prediction failed')
        if seed in (0, 41):
            _require(_digest_puzzle(puzzle) == _digest_puzzle(generate_puzzle(seed)), 'Determinism failed')
    _require(len(signatures) == 42, 'Puzzle seeds collapsed')
    _require(len(structures) >= 150 and len(shapes) >= 25, 'Insufficient geometric variation')
    _require(len(palettes) >= 20, 'Insufficient role variation')
    _require(counts == {3, 4, 5} and repair_counts == {1, 2, 3}, 'Missing counts or correction complexity')
    _require({10, 11, 12, 13, 14, 15} <= sizes, 'Missing scale variation')
    bad = generate_puzzle(0, 2, 1)
    # Swap carrier cells, preserving the anchor, palette and all color counts.
    # Only the complete geometry check should reject this plausible bad label.
    out = bad['test'][0]['output']
    a = (0, 1)
    b = next((int(r), int(c)) for r, c in np.argwhere(out != out[a]) if (int(r), int(c)) != (0, 0))
    out[a], out[b] = int(out[b]), int(out[a])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output was accepted')
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), output_structures=len(structures),
                input_shapes=len(shapes), role_palettes=len(palettes), piece_counts=sorted(counts),
                square_sides=sorted(sizes), repair_counts=sorted(repair_counts))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
