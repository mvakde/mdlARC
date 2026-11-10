"""Shattered window mosaic: shape ballots constrain and color a rigid assembly.

Only standard library and NumPy. No files, examples, random state, or generation
metadata are consulted by solve(). Fixed semantic colors: 0 empty, 5 window,
9 anchor. Other colors are interchangeable carriers and ballot choices.
"""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from math import isqrt

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
COLORS = (1, 2, 3, 4, 6, 7, 8)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


@lru_cache(maxsize=2048)
def _variants(shape, reflection=True):
    result = set()
    for flip in range(2 if reflection else 1):
        current = _normalize((r, -c if flip else c) for r, c in shape)
        for _ in range(4):
            result.add(current)
            current = _normalize((c, -r) for r, c in current)
    return tuple(sorted(result))


def _signature(cells):
    return min(_variants(_normalize(cells)))


def _extent(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _components(mask, neighbors=EDGE):
    remaining = set((int(r), int(c)) for r, c in np.argwhere(mask))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), 'Grid bounds')
    _need(np.issubdtype(g.dtype, np.integer), 'Integer grids required')
    _need(bool(np.all((0 <= g) & (g <= 9))), 'Color bounds')
    return g.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _tight(cells, grid):
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    bottom, right = max(r for r, c in cells), max(c for r, c in cells)
    result = np.full((bottom-y+1, right-x+1), -1, dtype=np.int8)
    for r, c in cells:
        result[r-y, c-x] = grid[r, c]
    return result


def _read(input_grid):
    """Segmentation uses object texture, never an inferred color dictionary."""
    grid = _grid(input_grid)
    shards, votes, vote_raws, shard_cells = [], {}, {}, []
    for cells in _components(grid != 0, EIGHT):
        values = {int(grid[r, c]) for r, c in cells}
        if 5 in values:
            carrier = values - {5, 9}
            _need(len(carrier) == 1 and carrier <= set(COLORS), 'Shard carrier')
            _need(len(cells) >= 4, 'Shard too small')
            _need(len(_components(_tight(cells, grid) >= 0)) == 1, 'Edge-connected shards required')
            shards.append(_tight(cells, grid))
            shard_cells.append(cells)
        else:
            _need(len(values) == 1 and values <= set(COLORS), 'Monochromatic ballot required')
            _need(len(_components(_tight(cells, grid) >= 0)) == 1, 'Edge-connected ballot required')
            color = next(iter(values))
            shape = _signature(cells)
            votes.setdefault(shape, Counter())[color] += 1
            vote_raws.setdefault(shape, []).append((color, _normalize(cells)))
    _need(3 <= len(shards) <= 5, 'Three to five shards required')
    _need(int(np.sum(grid == 9)) == 1, 'One anchor required')
    carriers = [next(int(v) for v in np.unique(p) if v in COLORS) for p in shards]
    _need(len(set(carriers)) == len(carriers), 'Distinct shard carriers required')
    area = sum(int(np.sum(p >= 0)) for p in shards)
    side = isqrt(area)
    _need(side * side == area, 'Shard area must be square')
    _need(2 <= len(votes) <= 4, 'Two to four supported shape classes required')
    winners = {}
    for shape, ballot in votes.items():
        high = max(ballot.values())
        best = [c for c, n in ballot.items() if n == high]
        _need(len(best) == 1, 'Tied shape ballot')
        winners[shape] = best[0]
    return dict(grid=grid, shards=shards, side=side, votes=votes,
                winners=winners, vote_raws=vote_raws, shard_cells=shard_cells)


def _placements(piece, side, allow_rotations=True):
    """Each entry is a whole rigid colored object at a legal square position."""
    candidates = {}
    for k in range(4 if allow_rotations else 1):
        p = np.rot90(piece, k)
        h, w = p.shape
        if h > side or w > side:
            continue
        anchored = bool(np.any(p == 9))
        if anchored and p[0, 0] != 9:
            continue
        cells = tuple((int(r)*side+int(c), int(p[r, c]))
                      for r, c in np.argwhere(p >= 0))
        mask = sum(1 << pos for pos, value in cells)
        windows = sum(1 << pos for pos, value in cells if value == 5)
        offsets = (0,) if anchored else (r*side+c for r in range(side-h+1)
                                        for c in range(side-w+1))
        for shift in offsets:
            # Uniform carrier plus windows makes these masks lossless here.
            key = mask << shift, windows << shift
            candidates.setdefault(key, (key[0], key[1], shift, cells))
    return tuple(candidates.values())


def _assemblies(shards, side, catalogue, limit=2, allow_rotations=True, owner_records=None):
    """Enumerate anchored exact covers; the completed windows must match votes.

    The raw geometric-cover count is useful evidence: a unique valid mosaic can
    arise despite many geometrically possible square packings. No answer or
    construction partition is supplied to this search.
    """
    choices = [_placements(p, side, allow_rotations) for p in shards]
    anchor = [i for i, p in enumerate(shards) if np.any(p == 9)]
    if len(anchor) != 1 or any(not options for options in choices):
        return [], 0
    anchor = anchor[0]
    full = (1 << (side*side))-1
    placed, answers, answer_keys = [None]*len(shards), [], set()
    geometric_count = 0

    def visit(occupied, windows, remaining):
        nonlocal geometric_count
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            geometric_count += 1
            mask = np.fromiter(((windows >> i) & 1 for i in range(side*side)),
                               dtype=np.int8, count=side*side).reshape(side, side)
            if np.any(mask[[0, -1]]) or np.any(mask[:, [0, -1]]):
                return
            groups = _components(mask != 0)
            if {_signature(c) for c in groups} != catalogue:
                return
            raw = np.empty((side, side), dtype=np.int8)
            for _, _, shift, cells in placed:
                for pos, value in cells:
                    raw.flat[pos+shift] = value
            key = raw.tobytes()
            if key not in answer_keys:
                answer_keys.add(key)
                answers.append(raw)
                if owner_records is not None:
                    owners = np.empty(raw.shape, dtype=np.int8)
                    for i, (_, _, shift, cells) in enumerate(placed):
                        for pos, value in cells:
                            owners.flat[pos+shift] = i
                    owner_records.append(owners)
            return
        gap = full ^ occupied
        first = gap & -gap
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in choices[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    visit(occupied | candidate[0], windows | candidate[1], rest)
            placed[i] = None

    rest = tuple(i for i in range(len(shards)) if i != anchor)
    for candidate in choices[anchor]:
        placed[anchor] = candidate
        visit(candidate[0], candidate[1], rest)
    return answers, geometric_count


def _repair(raw, winners):
    answer = raw.copy()
    for cells in _components(raw == 5):
        color = winners[_signature(cells)]
        for r, c in cells:
            answer[r, c] = color
    return answer


def _decode(input_grid):
    facts = _read(input_grid)
    owners = []
    answers, geometric = _assemblies(facts['shards'], facts['side'], set(facts['votes']), owner_records=owners)
    _need(len(answers) == 1, 'Require exactly one catalogue-compatible rigid assembly')
    facts.update(raw=answers[0], geometric=geometric, owners=owners[0])
    return _repair(answers[0], facts['winners']), facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a unique repaired square from this input alone."""
    return _decode(input_grid)[0]


@lru_cache(maxsize=1)
def _shape_pool():
    """Enumerate connected free polyominoes, not a bank of complete puzzles."""
    level = {((0, 0),)}
    pool = set()
    for area in range(2, 8):
        following = set()
        for shape in sorted(level):
            cells = set(shape)
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in EDGE}-cells)
            for p in frontier:
                grown = _signature(cells | {p})
                if max(_extent(grown)) <= 5:
                    following.add(grown)
        level = following
        if area >= 4:
            pool.update(level)
    return tuple(sorted(pool))


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _partition(rng, side, count, style):
    """Recursive rectangular cuts, optionally converted to interlocking shards."""
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        eligible = [i for i, (r, c, h, w) in enumerate(rectangles)
                    if h*w >= 10 and max(h, w) >= 4]
        if not eligible:
            return None
        i = _pick(rng, eligible)
        r, c, h, w = rectangles.pop(i)
        options = [(0, x) for x in range(2, h-1)] + [(1, x) for x in range(2, w-1)]
        if not options:
            return None
        axis, x = _pick(rng, options)
        rectangles.extend([(r, c, x, w), (r+x, c, h-x, w)] if axis == 0 else
                          [(r, c, h, x), (r, c+x, h, w-x)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    if style == 'stepped':
        for _ in range(side*4):
            r, c = map(int, rng.integers(side, size=2))
            dr, dc = _pick(rng, EDGE)
            nr, nc = r+dr, c+dc
            if not (0 <= nr < side and 0 <= nc < side) or (r, c) == (0, 0):
                continue
            a, b = int(labels[r, c]), int(labels[nr, nc])
            if a == b or int(np.sum(labels == a)) <= 5:
                continue
            labels[r, c] = b
            if len(_components(labels == a)) != 1:
                labels[r, c] = a
    return labels


def _window_layout(rng, labels, shapes, repetitions):
    side = len(labels)
    count = int(labels.max())+1
    for _ in range(100):
        mask = np.zeros(labels.shape, dtype=bool)
        free = np.ones(labels.shape, dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        records = []
        order = sorted(shapes+repetitions, key=lambda s: -max(_extent(s)))
        for shape in order:
            # Canonical handedness lets reflection ballots be discriminating.
            oriented = _pick(rng, _variants(shape, False))
            h, w = _extent(oriented)
            available = _legal_origins(free, oriented, h, w)
            if not len(available):
                break
            y, x = map(int, _pick(rng, available))
            cells = tuple((y+r, x+c) for r, c in oriented)
            for r, c in cells:
                mask[r, c] = True
                free[r-1:r+2, c-1:c+2] = False
            records.append((shape, cells))
        if len(records) != len(order):
            continue
        if any(not mask[labels == i].any() or np.sum(~mask[labels == i]) < 3 for i in range(count)):
            continue
        # At least one visible glyph is genuinely distributed across a seam.
        if not any(len({int(labels[r, c]) for r, c in cells}) > 1 for _, cells in records):
            continue
        return mask, records
    return None


def _shards(labels, window_mask, carriers):
    raw = np.array(carriers, dtype=np.int8)[labels]
    raw[window_mask] = 5
    raw[0, 0] = 9
    pieces = [_tight(tuple(map(tuple, np.argwhere(labels == i))), raw) for i in range(len(carriers))]
    return raw, pieces


def _legal_origins(free, cells, height, width):
    rows, cols = free.shape[0]-height+1, free.shape[1]-width+1
    if rows <= 0 or cols <= 0:
        return []
    valid = np.ones((rows, cols), dtype=bool)
    for r, c in cells:
        valid &= free[r:r+rows, c:c+cols]
    return np.argwhere(valid)


def _scatter(rng, shards, tokens, teaching=False):
    rotated = []
    for piece in shards:
        # The marked teaching shard witnesses rotation; other shards and all
        # later examples may use any of the four orientations.
        low = 1 if teaching and np.any(piece == 9) else 0
        rotated.append(np.rot90(piece, int(rng.integers(low, 4))).copy())
    objects = rotated + tokens
    order = list(map(int, rng.permutation(len(objects))))
    order.sort(key=lambda i: (-max(objects[i].shape), -int(np.sum(objects[i] >= 0))))
    for attempt in range(18):
        if attempt < 6:
            height, width = map(int, rng.integers(23, 29, size=2))
        else:
            height = width = 30
        grid = np.zeros((height, width), dtype=np.int8)
        free = np.ones(grid.shape, dtype=bool)
        okay = True
        for i in order:
            p = objects[i]
            cells = tuple((int(r), int(c)) for r, c in np.argwhere(p >= 0))
            origins = _legal_origins(free, cells, *p.shape)
            if not len(origins):
                okay = False
                break
            y, x = map(int, origins[int(rng.integers(len(origins)))])
            for r, c in cells:
                rr, cc = y+r, x+c
                grid[rr, cc] = p[r, c]
                free[max(0, rr-1):rr+2, max(0, cc-1):cc+2] = False
        if okay:
            rr, cc = np.nonzero(grid)
            # Independent outer margins, including edge-touching objects.
            y = max(0, int(rr.min())-int(rng.integers(3)))
            x = max(0, int(cc.min())-int(rng.integers(3)))
            bottom = min(height, int(rr.max())+1+int(rng.integers(3)))
            right = min(width, int(cc.max())+1+int(rng.integers(3)))
            return grid[y:bottom, x:right].copy()
    return None


def _ballots(rng, shapes, first_winners=None, teaching=False):
    winners, tokens = {}, []
    chosen = list(map(int, rng.choice(COLORS, len(shapes), replace=False)))
    if first_winners is not None:
        # Reject a remembered shape-to-color association across demonstrations.
        for i, s in enumerate(shapes):
            options = [c for c in COLORS if c != first_winners[s] and c not in chosen[:i]]
            chosen[i] = _pick(rng, options)
    for i, shape in enumerate(shapes):
        winner = chosen[i]
        winners[shape] = winner
        if teaching and i == 0:
            counts = (3, 2, 2)
        else:
            counts = _pick(rng, ((1,), (2, 1), (3, 2), (2, 1, 1), (3, 2, 1), (4, 2, 1)))
        losing_colors = [c for c in COLORS if c != winner]
        rng.shuffle(losing_colors)
        for j, (color, count) in enumerate(zip([winner]+losing_colors, counts)):
            allowed = _variants(shape)
            if teaching and i == 0:
                # Winner ballots have opposite chirality to every matching
                # window; repeated losing colors are not a majority.
                base = _normalize((r, -c) for r, c in shape) if j == 0 else shape
                allowed = _variants(base, False)
            for _ in range(count):
                cells = _pick(rng, allowed)
                token = np.full(_extent(cells), -1, dtype=np.int8)
                for r, c in cells:
                    token[r, c] = color
                tokens.append(token)
    return winners, tokens


def _make_pair(rng, side, count, shapes, repeats, style, teaching,
               first_winners=None, require_ambiguity=False):
    failures = Counter()
    for _ in range(160):
        labels = _partition(rng, side, count, style)
        if labels is None:
            failures['partition'] += 1
            continue
        layout = _window_layout(rng, labels, shapes, repeats)
        if layout is None:
            failures['windows'] += 1
            continue
        mask, records = layout
        carriers = list(map(int, rng.choice(COLORS, count, replace=False)))
        raw, pieces = _shards(labels, mask, carriers)
        candidates, geometric = _assemblies(pieces, side, set(shapes))
        if len(candidates) != 1 or not np.array_equal(candidates[0], raw):
            failures['uniqueness'] += 1
            continue
        if require_ambiguity and geometric < 2:
            failures['ambiguity'] += 1
            continue
        winners, tokens = _ballots(rng, shapes, first_winners, teaching)
        inp = _scatter(rng, pieces, tokens, teaching)
        if inp is not None:
            return {'input': inp, 'output': _repair(raw, winners)}, winners
        failures['scatter'] += 1
    raise RuntimeError(f'Bounded construction exhausted: side={side}, shards={count}, shapes={shapes}, repeats={repeats}, style={style}, failures={dict(failures)}')


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _shape_pool()
    chiral = [s for s in pool if 4 <= len(s) <= 5 and max(_extent(s)) <= 3
              and _normalize((r, -c) for r, c in s) not in _variants(s, False)
              and any(t != s and len(t) == len(s) and _extent(t) == _extent(s) for t in pool)]
    anchor = _pick(rng, chiral)
    companion = _pick(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                          and _extent(s) == _extent(anchor)])
    teaching_shapes = [anchor, companion]
    puzzle, seen, previous = {'train': [], 'test': []}, set(), None
    for split, total in (('train', num_train), ('test', num_test)):
        for i in range(total):
            teaching = split == 'train' and i < 2
            if teaching:
                shapes = teaching_shapes
                side = int(rng.integers(8+i, 10+i))
                count = 3+i
                repeats = [] if i == 0 else [_pick(rng, shapes)]
                style = 'cuts' if i == 0 else 'stepped'
            else:
                side = int(rng.integers(8, 11)) if split == 'train' else int(rng.integers(11, 13))
                count = int(rng.integers(3, 6))
                classes = 2 if side <= 9 else int(rng.integers(2, 4))
                eligible = [s for s in pool if len(s) <= (5 if side <= 9 else 7)
                            and max(_extent(s)) <= (3 if split == 'train' else 5)]
                shapes = [eligible[int(j)] for j in rng.choice(len(eligible), classes, replace=False)]
                repeat_count = (int(rng.integers(2)) if classes == 2 else 0) if split == 'train' else (int(rng.integers(1, 3)) if classes == 2 else 1)
                repeats = [_pick(rng, shapes) for _ in range(repeat_count)]
                if repeats and side < 9:
                    side = 9
                style = _pick(rng, ('cuts', 'stepped'))
            for _ in range(5):
                pair, winning = _make_pair(rng, side, count, shapes, repeats, style, teaching,
                                           previous if teaching and i == 1 else None,
                                           require_ambiguity=teaching and i == 0)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Distinct input search exhausted')
            seen.add(key)
            puzzle[split].append(pair)
            if teaching and i == 0:
                previous = winning
    return puzzle


def _facts(pair):
    prediction, facts = _decode(pair['input'])
    _need(np.array_equal(prediction, _grid(pair['output'])), 'Incorrect mosaic or repair')
    raw = facts['raw']
    # Retain one full cover from the input-only search. Multiple ownership
    # assignments with an identical colored assembly are harmless; choosing
    # placements independently here could accidentally overlap pale cells.
    owners = facts['owners']
    _need(bool(np.all(owners >= 0)), 'Ownership coverage')
    windows = _components(raw == 5)
    crossing = sum(len({int(owners[r, c]) for r, c in cells}) > 1 for cells in windows)
    _need(crossing >= 1, 'At least one window must cross shard seams')
    _need(raw[0, 0] == 9 and not np.any(prediction == 5) and not np.any(prediction == 0), 'Color-role preservation')
    for color in COLORS + (9,):
        _need(np.sum(raw == color) == np.sum(prediction[raw != 5] == color), 'Non-window cells changed')
    nonrect = sum(not np.all(p >= 0) for p in facts['shards'])
    reflection_witness = False
    for cells in windows:
        signature = _signature(cells)
        rotation_counts = Counter(color for color, s in facts['vote_raws'][signature]
                                  if s in _variants(_normalize(cells), False))
        high = max(rotation_counts.values(), default=0)
        best = [c for c, n in rotation_counts.items() if n == high]
        reflection_witness |= best != [facts['winners'][signature]]
    # A local-fragment filling shortcut is compared in the correct assembly:
    # unmatched local fragments remain 5, so success cannot be due to a crop.
    fragment_result = raw.copy()
    for owner in range(len(facts['shards'])):
        for cells in _components((raw == 5) & (owners == owner)):
            color = facts['winners'].get(_signature(cells), 5)
            for r, c in cells:
                fragment_result[r, c] = color
    facts.update(windows=len(windows), crossing=crossing, nonrect=nonrect,
                 reflection=reflection_witness,
                 fragment_failure=not np.array_equal(fragment_result, prediction),
                 owners=owners)
    return facts


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'Split lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            facts = _facts({'input': inp, 'output': out})
            _need(facts['fragment_failure'], 'Local fragment shortcut must fail')
            analyses[split].append(facts)
    training = analyses['train']
    _need(any(f['geometric'] > 1 for f in training), 'Teach catalogue-constrained assembly')
    _need(any(f['reflection'] for f in training), 'Teach reflected ballot equivalence')
    _need(any(len({len(s) for s in f['votes']}) < len(f['votes'])
              and len(set(f['winners'].values())) > 1 for f in training), 'Teach shape rather than area grouping')
    _need(any(any(len(v) >= 3 and max(v.values())*2 <= sum(v.values())
                       for v in f['votes'].values()) for f in training), 'Teach non-majority plurality')
    history = {}
    for f in training:
        for s, color in f['winners'].items():
            history.setdefault(s, set()).add(color)
    _need(any(len(colors) > 1 for colors in history.values()), 'Reject fixed shape/color dictionary')
    _need(any(f['windows'] > len(f['votes']) for f in training), 'Teach repeated windows')


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 puzzles streamed one at a time; no payload files or trace storage."""
    signatures, vocabulary, input_shapes, output_sides = set(), set(), set(), set()
    counts, ballots, layouts, structures = set(), set(), set(), set()
    reflected = crossing = ambiguous = irregular = repeated = novel = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        train_vocabulary = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _facts(pair)
                vocabulary.update(f['votes'])
                input_shapes.add(np.asarray(pair['input']).shape)
                output_sides.add(f['side'])
                counts.add(len(f['shards']))
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in f['votes'].values())
                layouts.add(tuple(sorted(int(np.sum(p >= 0)) for p in f['shards'])))
                # Geometry measurement includes raw masks, independently of color.
                structures.add((f['raw'].shape, (f['raw'] == 5).tobytes(), f['owners'].tobytes()))
                reflected += f['reflection']
                crossing += f['crossing']
                ambiguous += f['geometric'] > 1
                irregular += f['nonrect'] > 0
                repeated += f['windows'] > len(f['votes'])
                if split == 'train':
                    train_vocabulary.update(f['votes'])
                else:
                    novel += bool(set(f['votes'])-train_vocabulary)
    _need(len(signatures) == 42, 'Seed diversity')
    _need(len(vocabulary) >= 40, 'Geometric vocabulary diversity')
    _need(len(input_shapes) >= 15 and len(output_sides) >= 5, 'Dimension diversity')
    _need(counts == {3, 4, 5} and len(layouts) >= 80, 'Partition diversity')
    _need(len(ballots) >= 6 and (3, 2, 2) in ballots, 'Ballot diversity')
    _need(len(structures) >= 200, 'Structural variation')
    _need(reflected >= 84 and crossing >= 210 and ambiguous >= 42, 'Inherited-concept witnesses')
    _need(irregular >= 42 and repeated >= 84 and novel >= 30, 'Connected variation')
    a = generate_puzzle(317, 2, 1)
    _need(_puzzle_digest(a) == _puzzle_digest(generate_puzzle(317, 2, 1)), 'Determinism')
    _need(_puzzle_digest(a) != _puzzle_digest(generate_puzzle(318, 2, 1)), 'Different seeds')
    for nt, ne in ((10, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(97, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal count accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect output accepted')
    # Turn the taught 3-2-2 ballot into 2-2-2-1, using one recolored
    # isolated token. This must fail from input rather than invent a tie rule.
    tied = a['train'][0]['input'].copy()
    f = _read(tied)
    shape = next(s for s, v in f['votes'].items() if sorted(v.values()) == [2, 2, 3])
    winner = f['winners'][shape]
    unused = next(c for c in COLORS if c not in f['votes'][shape])
    for cells in _components(tied != 0, EIGHT):
        if {int(tied[r, c]) for r, c in cells} == {winner} and _signature(cells) == shape:
            for r, c in cells:
                tied[r, c] = unused
            break
    try:
        solve(tied)
    except AssertionError as exc:
        _need('Tied shape ballot' in str(exc), 'Tie rejected for wrong reason')
    else:
        raise AssertionError('Tied input accepted')
    return dict(seeds=42, pairs=210, vocabulary=len(vocabulary), input_shapes=len(input_shapes),
                output_sides=sorted(output_sides), partition_area_patterns=len(layouts),
                ballot_patterns=len(ballots), structures=len(structures), ambiguous=ambiguous,
                irregular=irregular, repeated=repeated, novel_tests=novel)
