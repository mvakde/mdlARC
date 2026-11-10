"""Programmable spillways: cyclic gate control coupled to conserved liquid.

Only the visible input is used by solve.  No parent generator is imported.
The sampler's geometry records are used to construct labels, then discarded.
"""
from __future__ import annotations

from collections import deque
import numpy as np


HEADER = 4


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _decode(value):
    """Read the key, program, alternating cups, gates, and floor relief."""
    g = _grid(value)
    h, w = g.shape
    if h < 12 or w < 12:
        raise ValueError("the spillway and its key do not fit")
    bg, wall, liquid = int(g[0, 0]), int(g[0, w-3]), int(g[1, w-2])
    if len({bg, wall, liquid}) != 3:
        raise ValueError("key roles must be distinct")
    key = g[:3, w-3:].copy()
    key[1, 1] = wall
    if not np.all(key == wall):
        raise ValueError("liquid key must be a hollow 3 by 3 wall box")
    expected_header = np.full((HEADER, w), bg, dtype=np.int8)
    expected_header[:3, w-3:] = wall
    expected_header[1, w-2] = liquid
    word, origins = [], []
    c = 1
    while c < w-3:
        color = int(g[1, c])
        if color == bg:
            c += 1
            continue
        if (color in (wall, liquid) or c+1 >= w-3
                or not np.all(g[1:3, c:c+2] == color)):
            raise ValueError("malformed program swatch")
        if origins and c-origins[-1] < 3:
            raise ValueError("program entries need a visible gap")
        word.append(color)
        origins.append(c)
        expected_header[1:3, c:c+2] = color
        c += 2
    if (not 2 <= len(word) <= 6 or not 2 <= len(set(word)) <= 3
            or not np.array_equal(g[:HEADER], expected_header)):
        raise ValueError("header does not uniquely encode a program")
    controls = set(word)
    if not set(map(int, np.unique(g))) <= controls | {bg, wall, liquid}:
        raise ValueError("unexplained color")

    # Floors are long solid runs anchored to a canvas side.  Vertical walls
    # and interior relief do not touch that side in a long horizontal run.
    floors = []
    for r in range(HEADER, h):
        row = g[r]
        if np.all(row == wall):
            floors.append((r, "sink", None))
            continue
        for side in ("left", "right"):
            scan = row if side == "left" else row[::-1]
            run = 0
            while run < w and scan[run] == wall:
                run += 1
            if run > w//2:
                boundary = run-1 if side == "left" else w-run
                floors.append((r, side, boundary))
    if (len(floors) < 2 or floors[-1] != (h-1, "sink", None)
            or any(side == "sink" for _, side, _ in floors[:-1])):
        raise ValueError("expected alternating cups followed by a bottom sink")
    cups, top, previous_side = [], HEADER, None
    source = np.zeros(g.shape, bool)
    for i, (floor, side, boundary) in enumerate(floors):
        if floor-top < 2:
            raise ValueError("chamber too shallow")
        region = np.zeros(g.shape, bool)
        gates = {}
        if side == "sink":
            region[top:floor, :] = True
            # The sink can have relief, but no colored gates or liquid source.
            if np.any(np.isin(g[top:floor], list(controls))):
                raise ValueError("sink has a gate")
        else:
            if side == previous_side or not 1 <= boundary <= w-2:
                raise ValueError("cups must alternate their open gutter sides")
            columns = slice(0, boundary) if side == "left" else slice(boundary+1, w)
            outside = slice(boundary+1, w) if side == "left" else slice(0, boundary)
            region[top:floor, columns] = True
            if not np.all(g[top:floor+1, outside] == bg):
                raise ValueError("the outside gutter must be clear")
            for r in range(top, floor):
                color = int(g[r, boundary])
                if color in controls:
                    if color in gates:
                        raise ValueError("duplicate gate label within a cup")
                    gates[color] = (r, boundary)
                elif color != wall:
                    raise ValueError("a side wall has an unexplained hole")
            if set(gates) != controls:
                raise ValueError("each cup needs exactly one gate of each program color")
            if min(floor-r for r, _ in gates.values()) < 2:
                raise ValueError("each gate must retain at least one level")
            if np.any(np.isin(g[region], list(controls))):
                raise ValueError("gate labels belong only in the side wall")
            previous_side = side
        free = region & (g != wall)
        relief = region & (g == wall)
        # Family floor relief is at most one cell high, wholly interior.
        if np.any(relief[:floor-1]):
            raise ValueError("floor relief is too high")
        if i == 0:
            source = free
        cups.append(dict(top=top, floor=floor, side=side, boundary=boundary,
                         gates=gates, free=free, relief=relief))
        top = floor+1
    if (not np.any(g[HEADER:] == liquid)
            or np.any((g == liquid) & (np.arange(h)[:, None] >= HEADER) & ~source)):
        raise ValueError("all supply must lie inside the first cup")
    # No cells in the body may fall outside the parsed geometry.
    restored = np.full(g.shape, bg, dtype=np.int8)
    restored[:HEADER] = g[:HEADER]
    for cup in cups:
        f, side, b, t = cup['floor'], cup['side'], cup['boundary'], cup['top']
        if side == 'sink':
            restored[f, :] = wall
        else:
            cols = slice(0, b+1) if side == 'left' else slice(b, w)
            restored[f, cols] = wall
            restored[t:f, b] = wall
            for color, point in cup['gates'].items():
                restored[point] = color
        restored[cup['relief']] = wall
    restored[(g == liquid) & source] = liquid
    if not np.array_equal(restored, g):
        raise ValueError("unparsed terrain or annotation")
    return dict(grid=g, background=bg, wall=wall, liquid=liquid,
                word=tuple(word), origins=tuple(origins), cups=cups,
                volume=int(np.count_nonzero(g[HEADER:] == liquid)))


def _lowest_levels(free, volume, complete=True):
    """Fill complete horizontal levels; no lateral tie is needed in the family."""
    selected = np.zeros(free.shape, bool)
    remaining = int(volume)
    for r in range(free.shape[0]-1, -1, -1):
        columns = np.flatnonzero(free[r])
        if not len(columns):
            continue
        if remaining < len(columns):
            if complete:
                raise ValueError("supply would partially fill a horizontal level")
            selected[r, columns[:remaining]] = True
            remaining = 0
            break
        selected[r, columns] = True
        remaining -= len(columns)
        if not remaining:
            break
    if remaining:
        raise ValueError("supply exceeds available capacity")
    return selected


def _allocate(info, alternative=None):
    """Hydrostatic cup capacities, with the cyclic program deciding each rim."""
    g, bg, liquid = info['grid'], info['background'], info['liquid']
    out = g.copy()
    out[:HEADER] = bg
    out[HEADER:][out[HEADER:] == liquid] = bg
    remaining, visited = info['volume'], []
    word = info['word']
    if alternative == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    rows = np.arange(g.shape[0])[:, None]
    for i, cup in enumerate(info['cups']):
        if not remaining:
            break
        visited.append(i)
        if cup['side'] == 'sink':
            water = _lowest_levels(cup['free'], remaining, alternative is None)
            remaining = 0
        else:
            color = word[i % len(word)]
            if alternative == 'reset':
                color = word[0]
            elif alternative == 'no_wrap':
                color = word[min(i, len(word)-1)]
            if alternative in ('lowest', 'highest'):
                choose = max if alternative == 'lowest' else min
                gate = choose(cup['gates'].values(), key=lambda p: p[0])
            else:
                gate = cup['gates'][color]
            out[gate] = bg
            capacity_mask = cup['free'] & (rows > gate[0])
            capacity = int(capacity_mask.sum())
            amount = min(remaining, capacity)
            water = _lowest_levels(capacity_mask, amount, alternative is None)
            remaining -= amount
        out[water] = liquid
    if remaining:
        raise ValueError("liquid does not fit the cascade")
    return out, tuple(visited)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Open visited sluices and settle the conserved supply, from input alone."""
    info = _decode(input_grid)
    return _allocate(info)[0]


def _pattern(word):
    seen = {}
    return tuple(seen.setdefault(int(x), len(seen)) for x in word)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _evidence(info, out):
    correct, visited = _allocate(info)
    assert np.array_equal(correct, out)
    evidence = set()
    # A is required for the water locations themselves, not just gate erasure.
    for mode in ('deduplicate', 'reset', 'no_wrap', 'lowest', 'highest'):
        try:
            wrong, _ = _allocate(info, mode)
        except ValueError:
            # An infeasible alternate budget is informative, but does not
            # count as our stronger witness of changed water locations.
            pass
        else:
            if not np.array_equal(wrong == info['liquid'], out == info['liquid']):
                evidence.add(mode)
    cups, word = info['cups'], info['word']
    first = cups[0]
    gate = first['gates'][word[0]]
    cap = int(np.count_nonzero(first['free'] & (np.arange(out.shape[0])[:, None] > gate[0])))
    if len(visited) == 1 and info['volume'] < cap:
        evidence.add('contained')
    if len(visited) >= 3:
        evidence.add('overflow')
    if len([i for i in visited if cups[i]['side'] != 'sink']) > len(word):
        evidence.add('cycling')
    if np.any((info['grid'][HEADER:] == info['liquid']).sum(axis=0)
              != (out[HEADER:] == info['liquid']).sum(axis=0)):
        evidence.add('lateral')
    if any(cup['relief'].any() and np.any((out == info['liquid']) & cup['free']
            & (np.arange(out.shape[0])[:, None] < cup['floor']-1)) for cup in cups):
        evidence.add('relief')
    return evidence


REQUIRED_EVIDENCE = {'deduplicate', 'reset', 'no_wrap', 'lowest', 'highest',
                     'contained', 'overflow', 'cycling', 'lateral', 'relief'}


def _sample_word(rng, colors, length):
    word = list(colors) + list(map(int, rng.choice(colors, length-len(colors))))
    rng.shuffle(word)
    return tuple(word)


def _make_pair(rng, roles, mode):
    """Construct capacities and an exact-level budget, then place that supply."""
    bg, wall, liquid = roles
    for attempt in range(240):
        if mode == 'spill':
            n, k, relief, offsets, length = 4, 2, False, [2, 3], 3
        elif mode == 'contained':
            n, k, relief, offsets = int(rng.integers(2, 4)), 2, True, [3, 4]
            length = int(rng.integers(3, 6))
        elif mode == 'depth':
            n, k, relief, offsets = 5, 2, False, [2, 3]
            length = int(rng.integers(2, 7))
        elif mode == 'gap':
            n, k, relief = int(rng.integers(2, 4)), 2, bool(rng.integers(2))
            offsets = [3, 5] if relief else [2, 4]
            length = int(rng.integers(3, 7))
        else:
            n = int(rng.integers(2, 5))
            relief = bool(rng.integers(2)) and n <= 3
            k = int(rng.integers(2, 4)) if n <= 3 else 2
            offsets = list(range(3 if relief else 2, (3 if relief else 2)+k))
            length = int(rng.integers(k, 7))
        colors = tuple(map(int, rng.choice([c for c in range(10) if c not in roles], k, replace=False)))
        if mode == 'spill':
            indices = (0, 0, 1) if rng.integers(2) else (0, 1, 1)
            word = tuple(colors[j] for j in indices)
        else:
            word = _sample_word(rng, colors, length)
        w = int(rng.integers(max(16, 3*length+4), 31))
        sink_span = 3 if mode == 'depth' else int(rng.integers(3, 6))
        other_spans = [max(offsets)+1 for _ in range(n-1)]
        if mode not in ('depth', 'spill'):
            other_spans = [s+int(rng.integers(2)) for s in other_spans]
        max_first = 30-HEADER-sink_span-sum(other_spans)
        min_first = max(7, max(offsets)+2)
        if max_first < min_first:
            continue
        first_span = 7 if mode == 'depth' else int(rng.integers(min_first, max_first+1))
        spans = [first_span] + other_spans + [sink_span]
        h = HEADER+sum(spans)
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[:3, w-3:] = wall
        inp[1, w-2] = liquid
        # Swatch spacing and horizontal offset vary while reading stays left to right.
        slack = w-4-(3*length)
        left = 1+int(rng.integers(min(3, slack+1)))
        gaps = np.ones(length-1, dtype=int)
        extra = slack-(left-1)
        for j in rng.permutation(length-1):
            if extra and rng.integers(2):
                gaps[j] += 1
                extra -= 1
        column = left
        for j, color in enumerate(word):
            inp[1:3, column:column+2] = color
            if j < len(gaps):
                column += 2+int(gaps[j])
        first_side = 'left' if rng.integers(2) else 'right'
        top, records = HEADER, []
        for i, span in enumerate(spans):
            f = top+span-1
            free = np.zeros(inp.shape, bool)
            gates = {}
            if i == n:
                side, b = 'sink', None
                free[top:f, :] = True
                inp[f, :] = wall
            else:
                side = first_side if i % 2 == 0 else ('right' if first_side == 'left' else 'left')
                gutter = (1 if mode == 'depth' else int(rng.integers(1, 4))) if i == 0 else int(rng.integers(1, w//2))
                b = w-gutter-1 if side == 'left' else gutter
                # Floor interior occupies strictly more than half the width.
                if side == 'left':
                    free[top:f, :b] = True
                    inp[f, :b+1] = wall
                else:
                    free[top:f, b+1:] = True
                    inp[f, b:] = wall
                inp[top:f, b] = wall
                order = list(map(int, rng.permutation(colors)))
                if mode == 'contained' and i == 0:
                    # The held amount is below the chosen high outlet.
                    order.remove(word[0])
                    order.append(word[0])
                for d, color in zip(offsets, order):
                    gates[color] = (f-d, b)
                    inp[f-d, b] = color
            interior = np.flatnonzero(free[f-1])
            has_relief = relief if i < n else bool(rng.integers(2))
            if has_relief and len(interior) >= 6:
                sites = interior[2:-2]
                # One-cell raised patches change capacity and create joined pockets.
                count = int(rng.integers(1, max(2, len(sites)//3+1)))
                chosen = rng.choice(sites, count, replace=False)
                inp[f-1, chosen] = wall
                free[f-1, chosen] = False
            records.append(dict(top=top, floor=f, side=side, boundary=b,
                                gates=gates, free=free))
            top = f+1
        finish = 0 if mode == 'contained' else n if mode in ('spill', 'depth') else int(rng.integers(1, n+1))
        target = np.zeros(inp.shape, bool)
        opened = []
        rr = np.arange(h)[:, None]
        for i in range(finish+1):
            rec = records[i]
            if i < n:
                gate = rec['gates'][word[i % length]]
                opened.append(gate)
                available = rec['free'] & (rr > gate[0])
                if i < finish:
                    target |= available
                    continue
                max_levels = rec['floor']-gate[0]-1
            else:
                available = rec['free']
                max_levels = spans[-1]-1
            has_pier = np.any(inp[rec['floor']-1][rec['free'][rec['floor']-2]] == wall)
            min_levels = 2 if has_pier else 1
            if mode == 'contained':
                levels = 2
            else:
                levels = int(rng.integers(min_levels, max_levels+1))
            target |= available & (rr >= rec['floor']-levels)
        volume = int(target.sum())
        eligible = np.argwhere(records[0]['free'])
        if volume > len(eligible) or not volume:
            continue
        # Source shape is only a volume carrier; it does not encode a path.
        style = int(rng.integers(5))
        anchor = eligible[int(rng.integers(len(eligible)))]
        dr, dc = np.abs(eligible[:, 0]-anchor[0]), np.abs(eligible[:, 1]-anchor[1])
        jitter = rng.random(len(eligible))
        if style == 0:
            score = dr+dc+jitter
        elif style == 1:
            score = jitter
        elif style == 2:
            score = 3*eligible[:, 0]+dc*.1+jitter
        elif style == 3:
            score = 3*dc+eligible[:, 0]*.2+jitter
        else:
            score = (~target[eligible[:, 0], eligible[:, 1]])*3+jitter
        chosen = eligible[np.argsort(score)[:volume]]
        if finish == 0 and np.all(target[chosen[:, 0], chosen[:, 1]]):
            outside = eligible[~target[eligible[:, 0], eligible[:, 1]]]
            chosen[-1] = outside[int(rng.integers(len(outside)))]
        out = inp.copy()
        out[:HEADER] = bg
        for gate in opened:
            out[gate] = bg
        out[target] = liquid
        inp[chosen[:, 0], chosen[:, 1]] = liquid
        pair = {'input': inp, 'output': out}
        info = _decode(inp)
        assert np.array_equal(_allocate(info)[0], out)
        if mode == 'spill' and not {'deduplicate', 'reset', 'no_wrap', 'lowest', 'highest', 'cycling'} <= _evidence(info, out):
            continue
        return pair
    raise RuntimeError("bounded spillway construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("legal counts are 2..10 train, 1..4 test, with train > test")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    shift = 'depth' if rng.integers(2) else 'gap'
    modes = ['contained', 'spill'] + ['general']*(int(num_train)-2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', modes), ('test', [shift]*int(num_test))):
        for mode in schedule:
            for _ in range(16):
                pair = _make_pair(rng, roles, mode)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _component(mask, start):
    seen = np.zeros(mask.shape, bool)
    seen[start] = True
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return seen


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, evidence = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            bg, wall, liquid = info['background'], info['wall'], info['liquid']
            roles.add((bg, wall, liquid))
            expected, visited = _allocate(info)
            assert np.array_equal(out, expected)
            assert np.array_equal(solve(inp.copy()), out)
            assert not np.array_equal(inp, out)
            assert info['volume'] == int(np.count_nonzero(out[HEADER:] == liquid))
            # Annotation removal is excluded from the physical volume.
            assert np.all(out[:HEADER] == bg)
            assert np.array_equal(inp[HEADER:] == wall, out[HEADER:] == wall)
            opened = {info['cups'][i]['gates'][info['word'][i % len(info['word'])]]
                      for i in visited if info['cups'][i]['side'] != 'sink'}
            for cup in info['cups']:
                for color, point in cup['gates'].items():
                    assert int(out[point]) == (bg if point in opened else color)
            before = np.argwhere(inp[HEADER:] == liquid)
            after = np.argwhere(out[HEADER:] == liquid)
            assert int(after[:, 0].sum()) > int(before[:, 0].sum())
            # Every final body is a complete, contained basin below its surface.
            water = (out == liquid)
            water[:HEADER] = False
            unseen = water.copy()
            solid = (out != bg) & (out != liquid)
            while unseen.any():
                start = tuple(map(int, np.argwhere(unseen)[0]))
                body = _component(water, start)
                surface = int(np.nonzero(body)[0].min())
                closure = _component(~solid & (np.arange(out.shape[0])[:, None] >= surface), start)
                assert np.array_equal(closure, body), "liquid body is unsupported or has an incomplete level"
                unseen[body] = False
            if split == 'train':
                evidence |= _evidence(info, out)
    assert len(roles) == 1, "background, wall, and liquid conventions change within a puzzle"
    assert REQUIRED_EVIDENCE <= evidence, REQUIRED_EVIDENCE-evidence


def _manual_regressions():
    """Hand-sized capacities: 14 in the first cup, 22 in the second."""
    for volume, second_levels, sink_levels in ((14, 0, 0), (25, 1, 0), (36, 2, 0), (52, 2, 1)):
        g = np.zeros((18, 16), dtype=np.int8)
        for c, color in ((1, 2), (4, 2), (7, 3)):
            g[1:3, c:c+2] = color
        g[:3, 13:] = 5
        g[1, 14] = 1
        g[4:10, 14] = 5
        g[7, 14], g[8, 14] = 3, 2
        g[10, :15] = 5
        g[11:14, 4] = 5
        g[11, 4], g[12, 4] = 2, 3
        g[14, 4:] = 5
        g[17, :] = 5
        positions = [(r, c) for r in range(4, 10) for c in range(14)]
        for point in positions[:volume]:
            g[point] = 1
        expected = g.copy()
        expected[:HEADER] = 0
        expected[4:][expected[4:] == 1] = 0
        expected[8, 14] = 0
        expected[9, :14] = 1
        if second_levels:
            expected[11, 4] = 0
            expected[14-second_levels:14, 5:] = 1
        if sink_levels:
            expected[16, :] = 1
        assert np.array_equal(solve(g), expected), volume
        # Rearranging supply within its cup must preserve the answer.
        reshaped = g.copy()
        reshaped[4:][reshaped[4:] == 1] = 0
        for point in positions[-volume:]:
            reshaped[point] = 1
        assert np.array_equal(solve(reshaped), expected)


def self_test():
    """Bounded 42-seed diversity, evidence, count, and capacity regressions."""
    _manual_regressions()
    puzzles, shapes, structures, words, roles, counts, volumes, offsets = set(), set(), set(), set(), set(), set(), set(), set()
    relief_cases, overflow_cases = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(np.asarray(p[k])) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        puzzles.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                out, visited = _allocate(info)
                assert np.array_equal(out, pair['output'])
                shapes.add(out.shape)
                words.add(_pattern(info['word']))
                roles.add((info['background'], info['wall'], info['liquid']))
                counts.add(len(info['cups'])-1)
                volumes.add(info['volume'])
                mapping = {}
                normalized = np.array([mapping.setdefault(int(c), len(mapping)) for c in pair['input'].flat], np.int8).reshape(out.shape)
                structures.add(_key(normalized))
                relief_cases += any(c['relief'].any() for c in info['cups'])
                overflow_cases += len(visited) >= 3
                for cup in info['cups'][:-1]:
                    offsets.add(tuple(sorted(cup['floor']-r for r, _ in cup['gates'].values())))
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 35 and len(words) >= 18 and len(roles) >= 35
    assert {2, 3, 4, 5} <= counts and len(volumes) >= 55
    assert len(offsets) >= 5 and relief_cases >= 60 and overflow_cases >= 65
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(20261014, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    first = generate_puzzle(927627577, 10, 4)
    second = generate_puzzle(927627577, 10, 4)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], second[s]) for k in ('input', 'output'))


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, and manual capacity regressions')
