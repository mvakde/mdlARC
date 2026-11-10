"""Contour printing: keyed ribbons travel on stripe-colored square contours.

Only the standard library and NumPy are used. No files, examples, or generator
state enter solve(). The catalog in each input makes the vocabulary visible.
"""
from __future__ import annotations

import numpy as np


def _arc_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _ring(n, d):
    """A clockwise cycle, with the northwest corner at index zero."""
    lo, hi = d, n - d - 1
    return ([(lo, c) for c in range(lo, hi + 1)]
            + [(r, hi) for r in range(lo + 1, hi + 1)]
            + [(hi, c) for c in range(hi - 1, lo - 1, -1)]
            + [(r, lo) for r in range(hi - 1, lo, -1)])


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        stack, cells = [q], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
        result.append(cells)
    return result


def _parse(value):
    a = _arc_grid(value)
    bg, divider = int(a[-1, 0]), int(a[-1, 2])
    if bg == divider:
        raise ValueError("missing divider")
    bars = np.flatnonzero(np.all(a == divider, axis=1))
    if len(bars) != 1:
        raise ValueError("expected one horizontal divider")
    k = int(bars[0])
    n = a.shape[0] - k - 1
    if not (1 <= k <= 3 and 4 <= n <= 26 and n % 2 == 0
            and a.shape[1] == n + 3):
        raise ValueError("invalid catalog/arena geometry")
    book = {}
    for row in a[:k]:
        code = int(row[0])
        if code in book or code in (bg, divider) or row[1] != divider or row[2] != bg:
            raise ValueError("invalid catalog label")
        used = np.flatnonzero(row[3:] != bg)
        if len(used) == 0:
            raise ValueError("empty ribbon")
        length = int(used[-1]) + 1
        word = row[3:3 + length] == divider
        if (not 4 <= length <= min(8, n) or not word[0] or not word[1]
                or not word[-1] or word.all()
                or not np.all(np.isin(row[3:], (bg, divider)))):
            raise ValueError("invalid ribbon mask")
        book[code] = word.copy()
    bottom = a[k + 1:]
    layers = n // 2
    sidebar = bottom[:, :2]
    if (not np.all(bottom[:, 2] == divider)
            or not np.all(sidebar == sidebar[:, :1])
            or not np.all(sidebar[layers + 1:] == bg)):
        raise ValueError("invalid stripe ruler")
    code = sidebar[:layers + 1, 0].copy()
    inks = set(map(int, code))
    if code[-1] != code[-2] or inks.intersection({bg, divider} | set(book)):
        raise ValueError("ruler colors collide with structural roles")
    arena = bottom[:, 3:].copy()
    if not set(map(int, np.unique(arena))) <= {bg} | set(book) | inks:
        raise ValueError("unexplained arena color")
    records, rejected, bare = [], [], []
    for cells in _components(arena, bg):
        points = [(r, c, int(arena[r, c])) for r, c in cells if int(arena[r, c]) in book]
        if len(cells) == 1 and points:
            records.extend(points)
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    return bg, divider, book, code, arena, records, rejected, bare


def _predict(value, mode="normal"):
    bg, _, book, code, arena, records, rejected, _ = _parse(value)
    n = arena.shape[0]
    out = np.full(arena.shape, bg, dtype=np.int8)
    if mode == "copy_ink":
        out[np.isin(arena, code)] = arena[np.isin(arena, code)]
    if mode == "all_codes":
        records = records + rejected
    layers = code[:-1]
    if mode == "reverse":
        layers = layers[::-1]
    elif mode == "collapse":
        layers = layers[np.r_[True, layers[1:] != layers[:-1]]]
    for r, c, symbol in records:
        d = min(r, c, n - 1 - r, n - 1 - c)
        ink = int(code[0] if mode == "outer" else layers[min(d, len(layers) - 1)])
        path = _ring(n, d)
        t = path.index((r, c))
        if mode == "origin":
            t = 0
        mask = book[symbol]
        if mode == "solid":
            mask = np.ones(len(mask), dtype=bool)
        if mode == "full":
            cells = path
        elif mode == "straight":
            cells = [(r, c + j) for j, bit in enumerate(mask) if bit and c + j < n]
        else:
            cells = [path[(t - j if mode == "counterclockwise" else t + j) % len(path)] for j, bit in enumerate(mask)
                     if bit and (mode != "nowrap" or t + j < len(path))]
        for q in cells:
            out[q] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the catalog and ruler, then print singleton symbols on their contours."""
    return _predict(input_grid)


def _walk(n, start, length):
    """Alternative traversal: turn at corners without indexing a stored cycle."""
    r, c = start
    d = min(r, c, n - 1 - r, n - 1 - c)
    hi = n - d - 1
    result = []
    for _ in range(length):
        result.append((r, c))
        if r == d and c < hi:
            c += 1
        elif c == hi and r < hi:
            r += 1
        elif r == hi and c > d:
            c -= 1
        else:
            r -= 1
    return result


def _render(n, bg, book, layers, records):
    out = np.full((n, n), bg, dtype=np.int8)
    for r, c, symbol in records:
        depth = min(r, c, n - r - 1, n - c - 1)
        for q, bit in zip(_walk(n, (r, c), len(book[symbol])), book[symbol]):
            if bit:
                out[q] = layers[depth]
    return out


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _free(arena, cells, bg):
    n = len(arena)
    return all(0 <= r < n and 0 <= c < n
               and np.all(arena[max(0, r - 1):min(n, r + 2),
                                   max(0, c - 1):min(n, c + 2)] == bg)
               for r, c in cells)


def _sample_book(rng, symbols):
    book, signatures = {}, set()
    for i, symbol in enumerate(symbols):
        for _ in range(100):
            length = int(rng.integers(4, 9))
            mask = rng.random(length) < rng.uniform(.35, .8)
            mask[:2] = mask[-1] = True
            mask[int(rng.integers(2, length - 1))] = False
            sig = tuple(map(bool, mask))
            if sig not in signatures and (i != 0 or sig != sig[::-1]):
                break
        else:
            # These distinct asymmetric words also provide a bounded fallback.
            mask = np.array([1, 1] + [0] * (i + 1) + [1], dtype=bool)
            sig = tuple(mask)
        book[int(symbol)] = mask
        signatures.add(sig)
    return book


def _sample_layers(rng, length, inks, demonstration):
    palette = list(map(int, rng.permutation(inks)))
    if demonstration:
        # A repeat followed by a color return defeats run compression; a
        # different terminal color makes forward and reverse order observable.
        layers = [palette[0], palette[1], palette[1], palette[0]] + palette[2:]
        while len(layers) < length - 1:
            layers.append(int(rng.choice(palette)))
        layers.append(palette[1])
    else:
        count = int(rng.integers(1, min(len(palette), length) + 1))
        chosen = palette[:count]
        layers = chosen + list(map(int, rng.choice(chosen, length - count)))
        rng.shuffle(layers)
    return np.asarray(layers, dtype=np.int8)


def _assemble(bg, divider, book, layers, arena):
    n, k = len(arena), len(book)
    inp = np.full((n + k + 1, n + 3), bg, dtype=np.int8)
    for i, (symbol, word) in enumerate(book.items()):
        inp[i, 0:2] = (symbol, divider)
        inp[i, 3:3 + len(word)][word] = divider
    inp[k, :] = divider
    inp[k + 1:, 2] = divider
    ruler = np.r_[layers, layers[-1]]
    inp[k + 1:k + 2 + len(layers), :2] = ruler[:, None]
    inp[k + 1:, 3:] = arena
    return inp


def _add_attached(rng, arena, bg, book, layers, records):
    """Source-style lower-right attachments, always visibly connected to ink."""
    n = len(arena)
    locations = [(r, c) for r in range(n - 2) for c in range(n - 2)]
    rng.shuffle(locations)
    desired = _render(n, bg, book, layers, records)
    symbols = list(book)
    for r, c in locations:
        size = 3 if r + 3 < n and c + 3 < n and rng.random() < .5 else 2
        shape = int(rng.integers(3))
        cells = [(r + a, c + b) for a in range(size) for b in range(size)
                 if shape == 0 or a in (0, size - 1) or b in (0, size - 1)]
        if shape == 2:
            cells = [(r + a, c + b) for a in range(size) for b in range(size)
                     if a == size - 1 or b == size - 1]
        point = (r + size, c + size)
        if not _free(arena, cells + [point], bg):
            continue
        symbol = int(rng.choice(symbols))
        extra = _render(n, bg, book, layers, records + [(*point, symbol)])
        if np.array_equal(extra, desired):
            continue  # The demonstration must visibly defeat stamp-every-code.
        ink = int(rng.choice(layers))
        for q in cells:
            arena[q] = ink
        arena[point] = symbol
        return True
    return False


def _add_bare(rng, arena, bg, book, layers, records):
    n = len(arena)
    locations = list(map(tuple, rng.permutation(np.argwhere(arena == bg))))
    symbols = list(book)
    desired = _render(n, bg, book, layers, records)
    for r, c in locations:
        symbol = int(rng.choice(symbols))
        cells = [q for q, bit in zip(_walk(n, (int(r), int(c)), len(book[symbol])), book[symbol]) if bit]
        if _free(arena, cells, bg):
            d = min(r, c, n - 1 - r, n - 1 - c)
            if all(desired[q] == layers[d] for q in cells):
                continue
            for q in cells:
                arena[q] = layers[d]
            return True
    return False


def _make_pair(rng, bg, divider, book, inks, length, demonstration, sparse=False,
               artifacts=()):
    n = 2 * length
    symbols = list(book)
    layers = _sample_layers(rng, length, inks, demonstration)
    for _ in range(30):
        arena = np.full((n, n), bg, dtype=np.int8)
        records = []
        if demonstration:
            depths = list(range(length - 1, -1, -1))
        else:
            count = int(rng.integers(1, min(length, 4 if sparse else length) + 1))
            depths = sorted(map(int, rng.choice(length, count, replace=False)), reverse=True)
        for d in depths:
            path = _ring(n, d)
            candidates = list(map(int, rng.permutation(len(path))))
            if demonstration:
                s = n - 2 * d
                # Outer levels teach all four corner turns; the innermost
                # start teaches cyclic wrapping even with only two examples.
                preferred = (3 if d == length - 1 else
                             ((d % 4 + 1) * (s - 1) - 1) % len(path))
                candidates = [preferred] + [t for t in candidates if t != preferred]
            choices = [path[t] for t in candidates if _free(arena, [path[t]], bg)]
            if not choices:
                break
            r, c = choices[0]
            symbol = symbols[d] if demonstration and d < len(symbols) else symbols[0] if demonstration else int(rng.choice(symbols))
            arena[r, c] = symbol
            records.append((r, c, symbol))
        if len(records) != len(depths):
            continue
        extras = int(rng.integers(0, 3 if sparse else 8))
        if sparse:
            extras = min(extras, max(0, 5 - len(records)))
        candidates = list(map(tuple, rng.permutation(np.argwhere(arena == bg))))
        for r, c in candidates:
            if extras == 0:
                break
            if _free(arena, [(r, c)], bg):
                symbol = int(rng.choice(symbols))
                arena[r, c] = symbol
                records.append((int(r), int(c), symbol))
                extras -= 1
        if 'attached' in artifacts and not _add_attached(rng, arena, bg, book, layers, records):
            continue
        if 'attached' in artifacts and rng.random() < .5:
            _add_attached(rng, arena, bg, book, layers, records)
        if 'bare' in artifacts and not _add_bare(rng, arena, bg, book, layers, records):
            continue
        if 'bare' in artifacts and rng.random() < .5:
            _add_bare(rng, arena, bg, book, layers, records)
        inp = _assemble(bg, divider, book, layers, arena)
        out = _render(n, bg, book, layers, records)
        parsed = _parse(inp)
        if sorted(parsed[5]) != sorted(records):
            continue
        if demonstration and any(np.array_equal(out, _predict(inp, mode))
                                 for mode in ("outer", "reverse", "collapse", "full", "straight", "nowrap",
                                              "counterclockwise", "origin", "solid")):
            continue
        return {"input": inp, "output": out}
    raise AssertionError("bounded contour/decoy construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    k = int(rng.integers(1, 4))
    bg, divider = palette[:2]
    book = _sample_book(rng, palette[2:2 + k])
    ink_count = int(rng.integers(2, min(5, 8 - k) + 1))
    inks = palette[2 + k:2 + k + ink_count]
    shift = int(rng.integers(3))  # larger, smaller, familiar but sparse
    upper = 10 if shift == 0 else 13 if shift == 1 else 11
    lower = 7 if shift == 1 else 4
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        training = i < num_train
        demo = training and i < 2
        if demo:
            length = int(rng.integers(max(6, lower, ink_count + 3), upper + 1))
            if i == 1:
                old = puzzle['train'][0]['output'].shape[0] // 2
                options = [v for v in range(max(6, lower, ink_count + 3), upper + 1) if v != old]
                length = int(rng.choice(options))
        elif training:
            length = int(rng.integers(lower, upper + 1))
        elif shift == 0:
            length = int(rng.integers(11, 14))
        elif shift == 1:
            length = int(rng.integers(4, 7))
        else:
            length = int(rng.choice([len(p['output']) // 2 for p in puzzle['train']]))
        artifacts = ({'attached', 'bare'} if training and i == 0 else
                     {kind for kind in ('attached', 'bare') if rng.random() < .4})
        for _ in range(20):
            pair = _make_pair(rng, bg, divider, book, inks, length, demo,
                              sparse=not training and shift == 2, artifacts=artifacts)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError("bounded uniqueness construction exhausted")
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _turns(n, book, records):
    result = set()
    for r, c, symbol in records:
        d = min(r, c, n - r - 1, n - c - 1)
        hi = n - d - 1
        corners = {(d, d): 'NW', (d, hi): 'NE', (hi, hi): 'SE', (hi, d): 'SW'}
        path = _walk(n, (r, c), len(book[symbol]))
        # First and last bits are ink, so an intermediate corner separates
        # visibly printed pixels on its two incident sides.
        result.update(corners[q] for q in path[1:-1] if q in corners)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, convention, taught, ink_sets, sizes, turns = set(), None, set(), {}, set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _arc_grid(pair['input']), _arc_grid(pair['output'])
            bg, divider, book, code, arena, live, rejected, bare = _parse(inp)
            signature = (bg, divider, tuple((c, tuple(map(bool, w))) for c, w in book.items()))
            assert len({tuple(w) for w in book.values()}) == len(book)
            if convention is None:
                convention = signature
            assert signature == convention, 'vocabulary or palette roles changed within a puzzle'
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert out.shape == arena.shape and live
            assert np.array_equal(out, solve(inp)), 'input-only prediction disagrees'
            # The alternative renderer moves through corners step by step.
            assert np.array_equal(out, _render(len(arena), bg, book, code[:-1], live))
            assert set(map(int, np.unique(out))) <= {bg} | set(map(int, code))
            for cells in _components(arena, bg):
                points = [q for q in cells if int(arena[q]) in book]
                assert len(points) <= 1, 'incidental contact between symbols'
                assert len(cells) == 1 or any(int(arena[q]) not in book for q in cells)
            if split == 'train':
                sizes.add(len(arena))
                turns.update(_turns(len(arena), book, live))
                for r, c, symbol in live:
                    d = min(r, c, len(arena) - 1 - r, len(arena) - 1 - c)
                    ink_sets.setdefault(symbol, set()).add(int(code[d]))
                    if len(_ring(len(arena), d)) >= len(book[symbol]):
                        taught.add(symbol)
    assert len(sizes) >= 2, 'training must demonstrate different code lengths'
    assert taught == set(book), 'every catalog entry needs a visible training invocation'
    assert any(len(v) >= 2 for v in ink_sets.values()), 'fixed code-to-ink shortcut survives'
    assert turns == {'NW', 'NE', 'SE', 'SW'}, 'all four contour turns need demonstrations'
    for mode in ('outer', 'reverse', 'collapse', 'full', 'straight', 'nowrap', 'all_codes',
                 'counterclockwise', 'origin', 'solid', 'copy_ink'):
        assert any(not np.array_equal(_predict(p['input'], mode), p['output'])
                   for p in puzzle['train']), f'training fails to distinguish {mode}'
    assert any(_parse(p['input'])[7] for p in puzzle['train']), 'bare ink erasure not demonstrated'


def self_test():
    """Bounded 42-seed diversity checks, count checks, and semantic mutations."""
    signatures, shapes, words, layer_codes, counts, palettes = set(), set(), set(), set(), set(), set()
    shifts, lengths, overlaps = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        train_n = [len(p['output']) for p in puzzle['train']]
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                bg, divider, book, code, arena, live, _, _ = _parse(pair['input'])
                n = len(arena)
                shapes.add(pair['input'].shape)
                words.update(tuple(map(bool, w)) for w in book.values())
                normalized = {}
                layer_codes.add(tuple(normalized.setdefault(int(c), len(normalized)) for c in code))
                counts.add(len(live))
                lengths.add(n)
                palettes.update(map(int, np.unique(pair['input'])))
                printed = []
                for r, c, symbol in live:
                    printed.extend(q for q, bit in zip(_walk(n, (r, c), len(book[symbol])), book[symbol]) if bit)
                overlaps += len(set(printed)) < len(printed)
                if split == 'test':
                    shifts.add('larger' if n > max(train_n) else 'smaller' if n < min(train_n) else 'familiar')
        again = generate_puzzle(seed)
        assert all(np.array_equal(p[key], q[key]) for s in puzzle
                   for p, q in zip(puzzle[s], again[s]) for key in ('input', 'output'))
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(words) >= 15 and len(layer_codes) >= 100 and len(counts) >= 10
    assert len(lengths) >= 8 and palettes == set(range(10))
    assert shifts == {'larger', 'smaller', 'familiar'} and overlaps >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Tiny legal family member: a four-cell word on a four-cell inner contour.
    tiny = _assemble(0, 9, {8: np.array([1, 1, 0, 1], bool)}, np.array([2, 3]),
                     np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 8, 0, 0], [0, 0, 0, 0]], dtype=np.int8))
    assert np.array_equal(solve(tiny), np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 3, 3, 0], [0, 0, 0, 0]], dtype=np.int8))
    p = generate_puzzle(0, 2, 1)
    query = p['test'][0]
    bg = _parse(query['input'])[0]
    non_bg = np.argwhere(query['output'] != bg)
    query['output'][tuple(non_bg[0])] = bg
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted missing printed ink')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, legal counts, contour and corruption checks')
