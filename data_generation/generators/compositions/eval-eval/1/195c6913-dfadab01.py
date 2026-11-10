"""Cyclic stencil walkers. All solving information is visible in each input.

No files, parent modules, random state, or training examples are read by solve.
The executable self-test streams its 42 seeds and retains only small summaries.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRECTIONS = ((0, 1), (-1, 0))
_ALTERNATIVES = frozenset({'point', 'box', 'anchor_fit', 'constant', 'dedup',
                          'reset', 'consume', 'first_wins', 'no_clipping'})
_REQUIRED = _ALTERNATIVES | {'shape_route', 'full_stamp', 'two_cycles'}


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
    return not remaining


def _read_input(value):
    """Parse the framed stencil key, ordered swatches, and terrain board."""
    a = _grid(value)
    H, W = a.shape
    assert H >= 19 and W >= 15
    bg = int(a[0, 0])
    assert np.all(a[0] == bg)
    # The tops of the dictionary cards are horizontal bars on row one.
    columns = [int(c) for c in np.flatnonzero(a[1] != bg)]
    runs = []
    for c in columns:
        if not runs or c != runs[-1][-1] + 1:
            runs.append([c])
        else:
            runs[-1].append(c)
    assert 2 <= len(runs) <= 3
    card_width = len(runs[0])
    assert all(len(run) == card_width for run in runs)
    w = card_width - 4
    assert 2 <= w <= 4 and runs[0][0] == 1
    assert all(b[0] == a0[-1] + 2 for a0, b in zip(runs, runs[1:]))
    frame = int(a[1, columns[0]])
    assert frame != bg and all(int(a[1, c]) == frame for c in columns)
    # A full-width bar separates the key from the board. Card sides end above it.
    dividers = [r for r in range(9, min(12, H)) if np.all(a[r] == frame)]
    assert len(dividers) == 1
    divider = dividers[0]
    h = divider - 7
    assert 2 <= h <= 4 and H - divider - 1 >= 8
    book, expected = {}, np.full((divider + 1, W), bg, dtype=np.int8)
    for run in runs:
        c = run[0]
        expected[1, c:c+card_width] = frame
        expected[h+2, c:c+card_width] = frame
        expected[1:h+3, c] = frame
        expected[1:h+3, c+card_width-1] = frame
        code = int(a[2, c+1])
        block = a[2:2+h, c+3:c+3+w]
        colors = set(map(int, np.unique(block))) - {bg}
        assert len(colors) == 1 and code != bg and code not in book
        ink = colors.pop()
        mask = block == ink
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        assert _connected(mask) and not mask.all()
        book[code] = (ink, mask.copy())
        expected[2, c+1] = code
        expected[2:2+h, c+3:c+3+w] = block
    program_row = h + 4
    seq = []
    c = 1
    while c + 1 < W and int(a[program_row, c]) != bg:
        color = int(a[program_row, c])
        assert color in book
        assert np.all(a[program_row:program_row+2, c:c+2] == color)
        expected[program_row:program_row+2, c:c+2] = color
        seq.append(color)
        c += 3
    assert 2 <= len(seq) <= 6 and set(seq) <= set(book)
    expected[divider] = frame
    assert np.array_equal(a[:divider+1], expected), 'malformed visible key'
    board = a[divider+1:]
    colors, counts = np.unique(board, return_counts=True)
    other = [(int(c0), int(n)) for c0, n in zip(colors, counts) if int(c0) != bg]
    assert len(other) == 2
    start_color, start_count = min(other, key=lambda p: p[1])
    assert start_count == 1
    terrain = next(c0 for c0, n in other if c0 != start_color)
    role_colors = [bg, terrain, frame, start_color] + list(book) + [v[0] for v in book.values()]
    assert len(set(role_colors)) == len(role_colors), 'palette roles collide'
    assert len({m.tobytes() for _, m in book.values()}) == len(book)
    start = tuple(map(int, np.argwhere(board == start_color)[0]))
    mask = (board == terrain) | (board == start_color)
    assert mask.any() and (~mask).any()
    assert start[1] == 0 or not mask[start[0], start[1]-1]
    return dict(grid=a, bg=bg, terrain=terrain, frame=frame, start_color=start_color,
                divider=divider, book=book, sequence=tuple(seq), start=start, mask=mask)


def _walk(mask, book, sequence, start, mode='correct'):
    """Monotone stamp placement with original-terrain collision tests.

    Diagnostic modes are intentionally wrong alternatives used by validation.
    They never select the production rule or influence solve.
    """
    H, W = mask.shape
    if mode == 'dedup':
        sequence = tuple(dict.fromkeys(sequence))
    offsets = {}
    for code, (ink, stencil) in book.items():
        if mode == 'point':
            stencil = np.ones((1, 1), dtype=bool)
        elif mode == 'box':
            stencil = np.ones(stencil.shape, dtype=bool)
        offsets[code] = tuple((int(r), int(c)) for r, c in np.argwhere(stencil))

    def inspect(p, code):
        r, c = p
        if not (0 <= r < H and 0 <= c < W):
            return None, ()
        if mode == 'no_clipping':
            h, w = book[code][1].shape
            if r+h > H or c+w > W:
                return False, ()
        support = {p}
        if mode != 'anchor_fit':
            support.update((r+dr, c+dc) for dr, dc in offsets[code]
                           if r+dr < H and c+dc < W)
        blocked = tuple(sorted(q for q in support if not mask[q]))
        return not blocked, blocked

    p, direction, phase = start, 0, 0
    pending = sequence[0]
    ok, _ = inspect(p, pending)
    if not ok:
        return None
    out = np.where(mask, -1, -2).astype(np.int8)
    anchors, codes, collisions, leg_lengths = [], [], set(), [0]
    painted = set()
    overlaps = 0
    clipped = 0
    for _ in range(H + W + 1):
        r, c = p
        anchors.append(p)
        codes.append(pending)
        leg_lengths[-1] += 1
        stencil = book[pending][1]
        clipped += int(r+stencil.shape[0] > H or c+stencil.shape[1] > W)
        for dr, dc in offsets[pending]:
            q = (r+dr, c+dc)
            if q[0] < H and q[1] < W:
                overlaps += int(q in painted)
                if mode != 'first_wins' or q not in painted:
                    out[q] = book[pending][0]
                painted.add(q)
        phase += 1
        pending = sequence[0 if mode == 'constant' else phase % len(sequence)]
        dr, dc = _DIRECTIONS[direction]
        proposed = (r+dr, c+dc)
        ok, blocked = inspect(proposed, pending)
        if ok is None:
            end = 'edge'
            break
        if ok:
            p = proposed
            continue
        collisions.update(blocked)
        if mode == 'consume':
            phase += 1
            pending = sequence[phase % len(sequence)]
        direction = 1 - direction
        if mode == 'reset':
            phase = 0
            pending = sequence[0]
        dr, dc = _DIRECTIONS[direction]
        proposed = (r+dr, c+dc)
        ok, blocked = inspect(proposed, pending)
        if ok is None:
            end = 'edge'
            break
        if not ok:
            collisions.update(blocked)
            end = 'blocked'
            break
        leg_lengths.append(0)
        p = proposed
    else:
        raise AssertionError('monotone walker failed to terminate')
    return dict(paint=out, anchors=tuple(anchors), codes=tuple(codes),
                collisions=frozenset(collisions), legs=tuple(leg_lengths), end=end,
                overlaps=overlaps, clipped=clipped)


def _render(info, record):
    assert record is not None, 'initial stencil does not fit'
    out = np.full(info['grid'].shape, info['bg'], dtype=np.int8)
    board = np.where(info['mask'], info['terrain'], info['bg']).astype(np.int8)
    foreground = record['paint'] >= 0
    board[foreground] = record['paint'][foreground]
    for p in record['collisions']:
        board[p] = info['frame']
    out[info['divider']+1:] = board
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input's visible dictionary, program and terrain alone."""
    info = _read_input(input_grid)
    record = _walk(info['mask'], info['book'], info['sequence'], info['start'])
    return _render(info, record)


def _evidence(info, correct=None):
    if correct is None:
        correct = _walk(info['mask'], info['book'], info['sequence'], info['start'])
    target = _render(info, correct)
    result = set()
    if correct['clipped'] < len(correct['anchors']):
        result.add('full_stamp')
    if len(correct['anchors']) >= 2*len(info['sequence']):
        result.add('two_cycles')
    for mode in sorted(_ALTERNATIVES):
        wrong = _walk(info['mask'], info['book'], info['sequence'], info['start'], mode)
        if wrong is None or not np.array_equal(target, _render(info, wrong)):
            result.add(mode)
        if mode == 'anchor_fit' and wrong is not None and wrong['anchors'] != correct['anchors']:
            result.add('shape_route')
    return result


def _sample_stencil(rng, h, w):
    # A monotone spine touches all four sides. Added cells preserve connectivity.
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    while r < h-1 or c < w-1:
        if r == h-1 or (c < w-1 and rng.random() < .5):
            c += 1
        else:
            r += 1
        mask[r, c] = True
    candidates = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    rng.shuffle(candidates)
    for r, c in candidates:
        if rng.random() < .30 and mask[max(0,r-1):r+2, max(0,c-1):c+2].any():
            mask[r, c] = True
    # Never sample a solid rectangle; that would remove the stencil-fit concept.
    if mask.all():
        mask[0, -1] = False
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _sample_word(rng, codes, length):
    word = list(codes) + [int(rng.choice(codes)) for _ in range(length-len(codes))]
    rng.shuffle(word)
    return tuple(word)


def _terrain(rng, H, W):
    n = min(H-2, int(rng.integers(3, 8)))
    knots = np.unique(np.r_[0, rng.choice(np.arange(1,H-1), n, replace=False), H-1])
    left = np.rint(np.interp(np.arange(H), knots,
                            rng.uniform(-W*.30, W*.32, len(knots)))).astype(int)
    right = np.rint(np.interp(np.arange(H), knots,
                             rng.uniform(W*.35, W*1.25, len(knots)))).astype(int)
    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, max(0,left[r]):min(W,max(left[r]+3,right[r]))] = True
    for _ in range(int(rng.integers(0, 6))):
        r = int(rng.integers(H)); c = int(rng.integers(W))
        dh = int(rng.integers(1, 5)); dw = int(rng.integers(1, 7))
        mask[r:r+dh, c:c+dw] = bool(rng.integers(2))
    return mask


def _make_input(mask, start, book, sequence, palette, W):
    bg, terrain, frame, start_color = palette[:4]
    h, w = next(iter(book.values()))[1].shape
    divider = h + 7
    a = np.full((divider+1+mask.shape[0], W), bg, dtype=np.int8)
    for i, (code, (ink, stencil)) in enumerate(book.items()):
        c = 1 + i*(w+5)
        a[1, c:c+w+4] = a[h+2, c:c+w+4] = frame
        a[1:h+3, c] = a[1:h+3, c+w+3] = frame
        a[2, c+1] = code
        a[2:2+h, c+3:c+3+w][stencil] = ink
    for i, code in enumerate(sequence):
        a[h+4:h+6, 1+3*i:3+3*i] = code
    a[divider] = frame
    a[divider+1:] = np.where(mask, terrain, bg)
    a[divider+1+start[0], start[1]] = start_color
    return a


def _make_pair(rng, book, palette, split, needed, shift):
    h, w = next(iter(book.values()))[1].shape
    codes = tuple(book)
    minimum_width = len(book)*(w+4) + len(book)-1 + 2
    max_height = 30-(h+8)
    best = None
    # At most 160 small boards, 8 starts each; no unbounded rejection sampling.
    for attempt in range(160):
        if split == 'test' and shift == 'size':
            H = int(rng.integers(max_height-1, max_height+1))
            W = int(rng.integers(max(minimum_width,28),31))
        else:
            upper = max_height-2 if shift == 'size' else max_height
            H = int(rng.integers(10, upper+1))
            W = int(rng.integers(minimum_width,31))
        if split == 'test' and shift == 'program':
            length = int(rng.integers(6 if len(codes)==3 else 5, 7))
        else:
            length = int(rng.integers(len(codes)+1, min(5,len(codes)+2)+1))
        W = max(W, 3*length)  # Every program swatch must remain fully visible.
        sequence = _sample_word(rng, codes, length)
        mask = _terrain(rng, H, W)
        if not (.22 <= mask.mean() <= .90):
            continue
        starts = [(r,c) for r in range(H//3,H) for c in range(W)
                  if mask[r,c] and (c == 0 or not mask[r,c-1])]
        rng.shuffle(starts)
        for start in starts[:8]:
            record = _walk(mask, book, sequence, start)
            if record is None or len(record['anchors']) < 8:
                continue
            if needed and (len(record['legs']) < 2 or len(record['anchors']) < 2*length):
                continue
            inp = _make_input(mask, start, book, sequence, palette, W)
            info = dict(grid=inp, mask=mask, start=start, book=book, sequence=sequence,
                        bg=palette[0], terrain=palette[1], frame=palette[2], divider=h+7)
            evidence = _evidence(info, record) if needed else set()
            score = len(evidence & needed)
            if best is None or score > best[0]:
                best = (score, inp, _render(info,record), evidence)
            if not needed or needed <= evidence:
                return {'input': inp, 'output': _render(info,record)}
    if best is not None:
        return {'input': best[1], 'output': best[2]}
    raise RuntimeError('bounded terrain sampling found no suitable start')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('pair counts must be integers')
    nt, nq = int(num_train), int(num_test)
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)))
    count = int(rng.integers(2,4))
    sizes = ((2,3),(3,2),(3,3),(3,4),(4,3),(4,4))
    h,w = sizes[int(rng.integers(len(sizes)))]
    book, used_masks = {}, set()
    for i in range(count):
        for _ in range(80):
            stencil = _sample_stencil(rng,h,w)
            if stencil.tobytes() not in used_masks and _connected(stencil):
                break
        else:
            raise RuntimeError('distinct stencil sampling exhausted')
        used_masks.add(stencil.tobytes())
        book[palette[4+i]] = (palette[4+count+i], stencil)
    shift = str(rng.choice(['size','program']))
    puzzle, seen, evidence = {'train': [], 'test': []}, set(), set()
    for split, n in (('train',nt),('test',nq)):
        for i in range(n):
            needed = _REQUIRED-evidence if split == 'train' else set()
            for _ in range(12):
                pair = _make_pair(rng, book, palette, split, needed, shift)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct input sampling exhausted')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                evidence |= _evidence(_read_input(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, conventions = set(), set(), None
    board_heights, word_lengths = {'train': [], 'test': []}, {'train': [], 'test': []}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            board_heights[split].append(info['mask'].shape[0])
            word_lengths[split].append(len(info['sequence']))
            book = info['book']
            contract = (info['bg'],info['terrain'],info['frame'],info['start_color'],
                        tuple((c,ink,_key(m)) for c,(ink,m) in book.items()))
            if conventions is None:
                conventions = contract
            assert contract == conventions, 'dictionary or palette changed within puzzle'
            record = _walk(info['mask'],book,info['sequence'],info['start'])
            assert record is not None and len(record['anchors']) >= 8
            assert len(set(record['anchors'])) == len(record['anchors'])
            assert sum(record['legs']) == len(record['anchors'])
            assert len(record['anchors']) <= sum(info['mask'].shape)-1
            assert all(not info['mask'][p] for p in record['collisions'])
            assert all(info['mask'][p] for p in record['anchors'])
            assert np.array_equal(out,_render(info,record)), 'input-derived output mismatch'
            # In production every successful mask pixel is on original terrain.
            assert np.all(info['mask'][record['paint'] >= 0])
            assert set(info['sequence']) == set(book), 'all visible dictionary entries used'
            if split == 'train':
                evidence |= _evidence(info,record)
    assert _REQUIRED <= evidence, f'training misses distinctions: {sorted(_REQUIRED-evidence)}'
    assert (max(board_heights['train']) < min(board_heights['test']) or
            max(word_lengths['train']) < min(word_lengths['test'])), 'no intended train/test shift'


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for p in puzzle[split]:
            for role in ('input','output'):
                a = np.asarray(p[role],dtype=np.uint8)
                digest.update(bytes(a.shape)); digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # A hollow stencil fits around a blocked centre; a solid box cannot fit.
    ring = np.ones((3,3),bool); ring[1,1] = False
    corner = np.array([[1,1,1],[1,0,0],[1,0,0]],bool)
    book = {4:(6,ring),5:(7,corner)}
    mask = np.ones((6,8),bool); mask[3,2] = False
    assert _walk(mask,book,(4,5,4),(2,1)) is not None
    assert _walk(mask,book,(4,5,4),(2,1),'box') is None
    # A horizontal canvas exit stops even when an upward continuation exists.
    full = np.ones((5,7),bool)
    rec = _walk(full,book,(4,5,4),(4,0))
    assert rec['anchors'] == tuple((4,c) for c in range(7))
    assert rec['end'] == 'edge' and not rec['collisions'] and rec['clipped'] == 7
    # Explicit first-fit failure, no phase consumption, and vertical exit.
    ledge = np.zeros((7,10),bool)
    ledge[3:,:5] = True; ledge[:4,:9] = True
    rec = _walk(ledge,book,(4,5,4),(5,0))
    assert rec is not None and len(rec['legs']) > 1 and rec['collisions']
    assert all(code == (4,5,4)[i%3] for i,code in enumerate(rec['codes']))


def self_test():
    """42 streaming seeds, count boundaries, deterministic and mutation checks."""
    _regressions()
    hashes, shapes, masks, words, leg_counts, end_types, palettes = (set() for _ in range(7))
    clipped = overlaps = bent = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_fingerprint(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                rec = _walk(info['mask'],info['book'],info['sequence'],info['start'])
                assert np.array_equal(solve(pair['input']),pair['output'])
                shapes.add(pair['input'].shape)
                masks.update(_key(m) for _,m in info['book'].values())
                names = {}; words.add(tuple(names.setdefault(c,len(names)) for c in info['sequence']))
                leg_counts.add(len(rec['legs'])); end_types.add(rec['end'])
                palettes.add((info['bg'],info['terrain'],info['frame'],info['start_color']))
                clipped += bool(rec['clipped']); overlaps += bool(rec['overlaps'])
                bent += len(rec['legs']) > 1
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(hashes) == 42 and len(shapes) >= 25 and len(masks) >= 25
    assert len(words) >= 10 and len(leg_counts) >= 3 and end_types == {'edge','blocked'}
    assert len(palettes) >= 40 and min(clipped,overlaps,bent) >= 10
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1337,nt,nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(2.0,1),(True,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The query label does not decide membership or fit: even one wrong pixel fails.
    mutated = generate_puzzle(99,2,1)
    query = mutated['test'][0]
    info = _read_input(query['input'])
    rec = _walk(info['mask'],info['book'],info['sequence'],info['start'])
    r,c = map(int,np.argwhere(rec['paint'] >= 0)[0])
    query['output'][info['divider']+1+r,c] = info['bg']
    try:
        validate_generated_puzzle(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('deleted paint accepted')
    return {'seeds':42,'pairs':210,'distinct_puzzles':len(hashes),'shapes':len(shapes),
            'stencils':len(masks),'word_patterns':len(words),'leg_counts':sorted(leg_counts),
            'clipped_pairs':clipped,'overlapping_pairs':overlaps,'bent_pairs':bent}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
