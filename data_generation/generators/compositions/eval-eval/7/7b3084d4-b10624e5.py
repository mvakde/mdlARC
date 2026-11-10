"""Chiral shared-joint jigsaws. No parent imports or generation state in solve."""
from __future__ import annotations

from itertools import product
from math import isqrt
import numpy as np


_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask, neighbors=_EIGHT):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    if not cells:
        return False
    stack = [cells.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in cells:
                cells.remove(p)
                stack.append(p)
    return not cells


def _tight(a):
    coords = np.argwhere(a >= 0)
    lo, hi = coords.min(0), coords.max(0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c + 1, w),
            (r + 1, h, 0, c), (r + 1, h, c + 1, w))


def _flip(a, reference, panel):
    # B's two axis reflections; rotations later make only their parity essential.
    if panel // 2 != reference // 2:
        a = a[::-1, :]
    if panel % 2 != reference % 2:
        a = a[:, ::-1]
    return a.copy()


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _check(all(int(a[r, c]) == bg for r, c in
               ((0, 0), (0, a.shape[1]-1), (a.shape[0]-1, 0),
                (a.shape[0]-1, a.shape[1]-1))), "corners must expose background")
    rows = [r for r in range(1, a.shape[0]-1)
            if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    cols = [c for c in range(1, a.shape[1]-1)
            if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    crosses = [(r, c) for r in rows for c in cols if a[r, 0] == a[0, c]]
    _check(len(crosses) == 1, "one full cross required")
    r, c = crosses[0]
    divider = int(a[r, c])
    bounds = _bounds(*a.shape, r, c)
    panels = [a[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    colors = [set(map(int, np.unique(p))) - {bg} for p in panels]
    common = set.intersection(*colors)
    _check(len(common) == 1, "one common joint identity required")
    joint = next(iter(common))
    _check(joint != divider, "joint must differ from divider")
    references = [i for i, s in enumerate(colors) if len(s) == 3]
    _check(len(references) == 1 and all(len(s) in (2, 3) for s in colors),
           "three plain fragments and one marked fragment required")
    reference = references[0]
    pieces, units, carriers = [], [], []
    marker = None
    for i, p in enumerate(panels):
        points = np.argwhere(p == joint)
        low, high = points.min(0), points.max(0) + 1
        side = int(high[0] - low[0])
        _check(side == high[1] - low[1] and len(points) == side * side,
               "joint must be one solid square per panel")
        raw = _tight(np.where(p != bg, p, -1).astype(np.int8))
        h, w = raw.shape
        _check(h % side == 0 and w % side == 0, "unaligned fragment extent")
        blocks = raw.reshape(h//side, side, w//side, side)
        small = blocks[:, 0, :, 0].copy()
        _check(np.all(blocks == small[:, None, :, None]), "scale boundaries must be exact")
        _check(np.count_nonzero(small == joint) == 1, "joint does not reduce to one cell")
        _check(_connected(small >= 0), "fragment must be corner-connected")
        other = colors[i] - {joint}
        if i == reference:
            single = [v for v in other if np.count_nonzero(small == v) == 1]
            _check(len(single) == 1, "unique one-unit marker required")
            marker = single[0]
            carrier = next(iter(other - {marker}))
        else:
            carrier = next(iter(other))
        _check(np.count_nonzero(small == carrier) >= 2, "carrier must occupy at least two units")
        pieces.append(small)
        units.append(side)
        carriers.append(carrier)
    _check(len(set(carriers)) == 4 and marker not in carriers,
           "carriers must be distinct and marker exclusive")
    area = sum(int(np.count_nonzero(p >= 0)) for p in pieces) - 3
    size = isqrt(area)
    _check(size * size == area and 1 <= size <= 30, "deduplicated area must be square")
    return dict(background=bg, divider=divider, joint=joint, marker=marker,
                reference=reference, units=units, pieces=pieces, side=size,
                bounds=bounds, carriers=carriers)


def _orientations(a, rotations):
    seen, result = set(), []
    for k in range(4 if rotations else 1):
        p = np.rot90(a, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _assemblies(data, *, reflections=True, rotations=True, fixed_reference=None, limit=2):
    """Enumerate colored answers; translations come only from the shared joint.

    Unlike disjoint exact cover, the four joint cells overlap and count once.
    The corner marker fixes each candidate's joint position and global frame.
    """
    n, joint, marker = data['side'], data['joint'], data['marker']
    ref = data['reference']
    flip_ref = ref if fixed_reference is None else fixed_reference
    pieces = [_flip(p, flip_ref, i) if reflections else p.copy()
              for i, p in enumerate(data['pieces'])]
    options = [_orientations(p, rotations) for p in pieces]
    answers, signatures = [], set()

    def placement(p, anchor_position):
        ar, ac = map(int, np.argwhere(p == joint)[0])
        dr, dc = anchor_position[0]-ar, anchor_position[1]-ac
        cells, mask = [], 0
        for r, c in np.argwhere(p >= 0):
            rr, cc = int(r)+dr, int(c)+dc
            if not (0 <= rr < n and 0 <= cc < n):
                return None
            if p[r, c] != joint:
                pos, color = rr*n+cc, int(p[r, c])
                mask |= 1 << pos
                cells.append((pos, color))
        return mask, cells

    for marked in options[ref]:
        mr, mc = map(int, np.argwhere(marked == marker)[0])
        jr, jc = map(int, np.argwhere(marked == joint)[0])
        anchor = jr-mr, jc-mc
        if not all(0 <= v < n for v in anchor):
            continue
        first = placement(marked, anchor)
        if first is None:
            continue
        full = ((1 << (n*n)) - 1) ^ (1 << (anchor[0]*n+anchor[1]))
        others = [i for i in range(4) if i != ref]
        candidates = [[q for p in options[i] if (q := placement(p, anchor)) is not None]
                      for i in others]
        if any(not q for q in candidates):
            continue
        out = np.full((n, n), -1, dtype=np.int8)
        out[anchor] = joint
        for pos, color in first[1]:
            out.flat[pos] = color

        def search(depth, mask):
            if limit is not None and len(answers) >= limit:
                return
            if depth == 3:
                if mask == full and out.tobytes() not in signatures:
                    signatures.add(out.tobytes())
                    answers.append(out.copy())
                return
            for new_mask, cells in candidates[depth]:
                if new_mask & mask:
                    continue
                for pos, color in cells:
                    out.flat[pos] = color
                search(depth+1, mask | new_mask)
                for pos, _ in cells:
                    out.flat[pos] = -1
        search(0, first[0])
        if limit is not None and len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique mosaic solely from input pixels and family conventions."""
    data = _decode(input_grid)
    answers = _assemblies(data)
    _check(len(answers) == 1, "input must have exactly one colored joint assembly")
    return answers[0]


def _partition(rng, n, joint, style):
    """Procedural frontier growth from four neighbors of a reserved joint."""
    labels = np.full((n, n), -1, dtype=np.int8)
    labels[joint] = 4
    near = [(joint[0]+dr, joint[1]+dc) for dr, dc in _EIGHT
            if 0 <= joint[0]+dr < n and 0 <= joint[1]+dc < n]
    if style == 0 and len([p for p in near if abs(p[0]-joint[0])+abs(p[1]-joint[1]) == 1]) == 4:
        near = [p for p in near if abs(p[0]-joint[0])+abs(p[1]-joint[1]) == 1]
    locked = style == 3 and all(0 < v < n-1 for v in joint)
    if locked:
        # A small symmetric fragment can coexist with irregular fragments.
        # Keep the two arms fixed; orientation duplicates are harmless.
        dr, dc = _EDGE[int(rng.integers(4))]
        bar = [(joint[0]+dr, joint[1]+dc), (joint[0]-dr, joint[1]-dc)]
        available = [p for p in near if p not in bar]
        groups = [bar] + [[available[int(i)]] for i in rng.choice(len(available), 3, replace=False)]
    else:
        groups = [[near[int(i)]] for i in rng.choice(len(near), 4, replace=False)]
    for i, group in enumerate(groups):
        for p in group:
            labels[p] = i
    weights = rng.uniform(.8, 1.2, 4) if style == 0 else np.exp(rng.uniform(-1.5, 1.5, 4))
    neighbors = []
    for r in range(n):
        for c in range(n):
            neighbors.append([(r+dr)*n+c+dc for dr, dc in _EIGHT
                              if 0 <= r+dr < n and 0 <= c+dc < n])
    frontier = [set() for _ in range(4)]
    sizes = [len(group) for group in groups]
    for i, group in enumerate(groups):
        for p in group:
            frontier[i].update(v for v in neighbors[p[0]*n+p[1]] if labels.flat[v] < 0)
    for _ in range(n*n-1-sum(sizes)):
        eligible = [i for i in range(4) if frontier[i] and not (locked and i == 0)]
        choice_weights = np.array([weights[i]/sizes[i]**.55 for i in eligible])
        i = int(rng.choice(eligible, p=choice_weights/choice_weights.sum()))
        choices = sorted(frontier[i])
        # Compact growth, mostly edge growth, and thin tendrils are all possible.
        counts = [sum(labels.flat[v] == i for v in neighbors[p]) for p in choices]
        if style == 2:
            target = min(counts)
            choices = [p for p, count in zip(choices, counts) if count == target]
        elif rng.random() < .7:
            edge = [p for p in choices if any(0 <= p//n+dr < n and 0 <= p % n+dc < n
                    and labels[p//n+dr, p % n+dc] == i for dr, dc in _EDGE)]
            if edge:
                choices = edge
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(pos)
        frontier[i].update(v for v in neighbors[pos] if labels.flat[v] < 0)
    return labels


def _scale_options(pieces, coarse, witness):
    options = []
    for units in product((2, 3) if coarse else (1, 2, 3), repeat=4):
        if not coarse and min(units) != 1:
            continue
        if witness and max(units) == 1:
            continue
        hs = [max(pieces[i].shape[0]*units[i]+2 for i in (0, 1)),
              max(pieces[i].shape[0]*units[i]+2 for i in (2, 3))]
        ws = [max(pieces[i].shape[1]*units[i]+2 for i in (0, 2)),
              max(pieces[i].shape[1]*units[i]+2 for i in (1, 3))]
        if sum(hs)+1 <= 30 and sum(ws)+1 <= 30:
            options.append((units, hs, ws))
    return options


def _spans(rng, minimum):
    extra = int(rng.integers(min(5, 29-sum(minimum))+1))
    first = int(rng.integers(extra+1))
    return minimum[0]+first, minimum[1]+extra-first


def _make_pair(rng, n, reference, edge_joint, coarse, witness, require_diagonal=False):
    for _ in range(600):
        if edge_joint:
            joint = (0, int(rng.integers(1, n-1)))
            for _ in range(int(rng.integers(4))):
                joint = n-1-joint[1], joint[0]
        else:
            joint = tuple(map(int, rng.integers(1, n-1, size=2)))
        labels = _partition(rng, n, joint, int(rng.integers(4)))
        marked_owner = int(labels[0, 0])
        if min(int(np.sum(labels == i)) for i in range(4)) < 2 or np.sum(labels == marked_owner) < 3:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, divider, anchor, marker = palette[:4]
        carriers = palette[4:8]
        if rng.random() < .4:
            carriers[int(rng.integers(4))] = divider
        output = np.empty((n, n), dtype=np.int8)
        for i, color in enumerate(carriers):
            output[labels == i] = color
        output[joint] = anchor
        output[0, 0] = marker
        owners = [i for i in range(4) if i != marked_owner]
        rng.shuffle(owners)
        owner_by_panel = []
        for p in range(4):
            owner_by_panel.append(marked_owner if p == reference else owners.pop())
        normalized = []
        for p, owner in enumerate(owner_by_panel):
            piece = _tight(np.where((labels == owner) | (labels == 4), output, -1))
            piece = np.rot90(piece, int(rng.integers(4))).copy()
            normalized.append(_flip(piece, reference, p))
        fake = dict(pieces=normalized, reference=reference, joint=anchor, marker=marker, side=n)
        answers = _assemblies(fake)
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        if require_diagonal and not any(not _connected(p >= 0, _EDGE) for p in normalized):
            continue
        if witness:
            # Fail rotation-free and mirror-free alternatives visibly, rather
            # than trusting sampled angles or a claimed chiral shape.
            if any(np.array_equal(q, output) for q in _assemblies(fake, rotations=False, limit=None)):
                continue
            if any(np.array_equal(q, output) for q in _assemblies(fake, reflections=False, limit=None)):
                continue
        choices = _scale_options(normalized, coarse, witness)
        if not choices:
            continue
        units, hs, ws = choices[int(rng.integers(len(choices)))]
        hs, ws = _spans(rng, hs), _spans(rng, ws)
        h, w = sum(hs)+1, sum(ws)+1
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[hs[0], :] = divider
        inp[:, ws[0]] = divider
        for p, (r0, r1, c0, c1) in enumerate(_bounds(h, w, hs[0], ws[0])):
            piece = np.repeat(np.repeat(normalized[p], units[p], axis=0), units[p], axis=1)
            y = int(rng.integers(1, r1-r0-piece.shape[0]))
            x = int(rng.integers(1, c1-c0-piece.shape[1]))
            region = inp[r0+y:r0+y+piece.shape[0], c0+x:c0+x+piece.shape[1]]
            region[piece >= 0] = piece[piece >= 0]
        return {'input': inp, 'output': output}
    raise RuntimeError('bounded procedural sampler could not find a unique joint jigsaw')


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    resolution_shift = bool(rng.integers(2))
    first_ref = int(rng.integers(4))
    second_ref = first_ref ^ int(rng.choice([1, 2]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = split == 'train' and i < 2
            reference = first_ref if witness and i == 0 else second_ref if witness else int(rng.integers(4))
            coarse = split == 'test' and resolution_shift
            n = int(rng.integers(4, 7)) if split == 'train' or coarse else int(rng.integers(7, 9))
            edge = (i == 1) if witness else bool(rng.random() < .35)
            for _ in range(12):
                pair = _make_pair(rng, n, reference, edge, coarse, witness,
                                  require_diagonal=witness and i == 1)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not obtain a distinct input')
        rng.shuffle(puzzle[split])
    return puzzle


def _features(pair):
    data = _decode(pair['input'])
    out = _grid(pair['output'])
    answers = _assemblies(data)
    _check(len(answers) == 1 and np.array_equal(answers[0], out), "incorrect or ambiguous assembly")
    _check(out[0, 0] == data['marker'] and np.sum(out == data['joint']) == 1,
           "marker orientation or joint coalescence wrong")
    _check(not np.any(out == data['background']), "no output gaps permitted")
    # Per-panel scale conservation, with exactly the three duplicate joints lost.
    for piece, carrier in zip(data['pieces'], data['carriers']):
        _check(np.sum(piece == carrier) == np.sum(out == carrier), "carrier area changed")
    _check(np.sum(out == data['marker']) == 1, "marker must reduce to one cell")
    data['rotation_witness'] = not any(np.array_equal(q, out) for q in
        _assemblies(data, rotations=False, limit=None))
    data['reflection_witness'] = not any(np.array_equal(q, out) for q in
        _assemblies(data, reflections=False, limit=None))
    data['fixed_ref_witness'] = not any(np.array_equal(q, out) for q in
        _assemblies(data, fixed_reference=0, limit=None))
    data['diagonal'] = any(not _connected(p >= 0, _EDGE) for p in data['pieces'])
    data['symmetric'] = any(len(_orientations(p, True)) < 4 for p in data['pieces'])
    jr, jc = map(int, np.argwhere(out == data['joint'])[0])
    data['edge_joint'] = jr in (0, len(out)-1) or jc in (0, len(out)-1)
    data['area_ratio'] = max(np.sum(p >= 0)-1 for p in data['pieces']) / min(np.sum(p >= 0)-1 for p in data['pieces'])
    return data


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            facts[split].append(_features({'input': inp, 'output': out}))
    witnesses = [f for f in facts['train'] if f['rotation_witness'] and f['reflection_witness']
                 and min(f['units']) == 1 and max(f['units']) > 1]
    _check(len(witnesses) >= 2, "two mixed-scale, rotation and chirality demonstrations required")
    _check(len({(f['reference']//2 + f['reference'] % 2) % 2 for f in witnesses}) == 2,
           "demonstrate both reference handedness classes")
    _check(any(f['fixed_ref_witness'] for f in witnesses), "reject absolute-quadrant handedness")
    _check(any(f['diagonal'] for f in facts['train']), "demonstrate corner connectivity")
    _check({f['edge_joint'] for f in facts['train']} == {False, True},
           "demonstrate an interior and an edge joint")
    _check(all(min(f['units']) == 1 for f in facts['train']),
           "training must include a unit-sized joint in each example")
    coarse_shift = all(min(f['units']) >= 2 for f in facts['test'])
    extent_shift = all(f['side'] > max(t['side'] for t in facts['train'])
                       and min(f['units']) == 1 for f in facts['test'])
    _check(coarse_shift or extent_shift, "missing resolution or extent shift")


def self_test():
    """Bounded semantic and structural audit of 42 seeds, plus negative checks."""
    signatures, structures, shapes, refs, sides, scales = set(), set(), set(), set(), set(), set()
    ratios, joint_types, shifts, symmetry_splits = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s]))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                _check(np.array_equal(solve(pair['input']), pair['output']), "solve disagreement")
                shapes.add(tuple(pair['input'].shape))
                refs.add(f['reference'])
                sides.add(f['side'])
                scales.update(f['units'])
                joint_types.add(f['edge_joint'])
                if f['symmetric']:
                    symmetry_splits.add(split)
                ratios.append(f['area_ratio'])
                roles, signature = {}, []
                for v in pair['output'].flat:
                    signature.append(roles.setdefault(int(v), len(roles)))
                structures.add((pair['output'].shape, tuple(signature)))
                local.append((split, f))
        if all(min(f['units']) >= 2 for s, f in local if s == 'test'):
            shifts.add('resolution')
        if min(f['side'] for s, f in local if s == 'test') > max(f['side'] for s, f in local if s == 'train'):
            shifts.add('extent')
        if seed < 2:
            again = generate_puzzle(seed)
            _check(all(_key(p[k]) == _key(q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ('input', 'output')), "nondeterminism")
    _check(len(signatures) == 42 and len(structures) >= 180, "insufficient geometry diversity")
    _check(len(shapes) >= 35 and refs == {0, 1, 2, 3}, "layout variation missing")
    _check(scales == {1, 2, 3} and sides == {4, 5, 6, 7, 8}, "scale variation missing")
    _check(joint_types == {False, True} and max(ratios) >= 5, "joint and area variation missing")
    _check(shifts == {'resolution', 'extent'}, "both test shifts required")
    _check(symmetry_splits == {'train', 'test'}, "symmetric fragments must occur in both splits")
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+a, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    sample = p['train'][0]
    for k in range(4):
        _check(np.array_equal(solve(np.rot90(sample['input'], k)), sample['output']),
               "rotating the entire scene changed the canonical mosaic")
    _check(np.array_equal(solve(9-sample['input']), 9-sample['output']),
           "solver depends on literal colors")
    data = _decode(sample['input'])
    damaged = sample['input'].copy()
    panel = next(i for i, side in enumerate(data['units']) if side > 1)
    r0, r1, c0, c1 = data['bounds'][panel]
    coords = np.argwhere(damaged[r0:r1, c0:c1] == data['carriers'][panel])
    row, col = map(int, coords[0])
    damaged[r0+row, c0+col] = data['background']
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonuniform reduction block accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds')
