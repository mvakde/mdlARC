"""Repair the obstacle scaffold, then paint the union of shortest gated routes.

Only standard-library/NumPy code is used.  The route color is always 4; the other
three roles are visible in each input.  No generation state is used by solve.
"""

from collections import deque
import hashlib
import json

import numpy as np


ROUTE = 4
STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 160415547, 290709483, 626236379, 893080888, 400776302,
              142807475, 851703712, 389972608, 668705780, 864782179,
              673848111, 332991654, 883747812, 26488781, 50882095,
              976197672, 248498033, 439124669, 324022116, 255430453,
              450852665, 553006953, 641346467, 105827173, 238736046,
              390436017, 896377352, 491915463, 804166641, 83456313,
              395141932, 940314250, 657937811, 481778160, 418981832,
              752134314, 214875462, 862987480, 26060357, 314382020,
              612721896)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    """Protect every cell in any orthogonal run of at least three cells."""
    result = np.zeros_like(mask)
    for m, dest in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            dest[:, k:k + triples.shape[1]] |= triples
    return result


def _gaps(mask):
    """Return positive-axis starts of bracketed, exactly three-cell breaks."""
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        if m.shape[1] < 5:
            continue
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repair_scaffold(mask):
    """Parent B's final visible criterion, evaluated simultaneously.

    A spur contains precisely two off-axis, non-strong cells, steps diagonally
    toward the opposite bracket, and has no third cell in that direction.  A
    straight wall must continue behind at least one bracket.  After filling,
    weak cells linked diagonally to another weak cell are removed.  Intact runs,
    including chamber shells, and retained repair brackets are protected.
    """
    strong = _strong(mask)
    additions = np.zeros_like(mask)
    anchors = np.zeros_like(mask)
    evidence = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and not _at(mask, ar + 3*sr, ac + 3*sc)):
                    matches.append((endpoint, side, spur))
        if len(matches) > 1:
            raise ValueError("ambiguous supported gap")
        if matches:
            for k in (1, 2, 3):
                additions[r + k*dr, c + k*dc] = True
            anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
            evidence.append(((r, c, dr, dc), matches[0]))
    weak = mask & ~_strong(mask | additions) & ~anchors
    diagonal = np.zeros_like(mask)
    for dc in (-1, 1):
        if dc == 1:
            linked = weak[:-1, :-1] & weak[1:, 1:]
            diagonal[:-1, :-1] |= linked
            diagonal[1:, 1:] |= linked
        else:
            linked = weak[:-1, 1:] & weak[1:, :-1]
            diagonal[:-1, 1:] |= linked
            diagonal[1:, :-1] |= linked
    repaired = (mask & ~diagonal) | additions
    return repaired, additions, mask & ~repaired, evidence


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or ROUTE in colors:
        raise ValueError("input needs three distinct non-route colors")
    background = int(colors[np.argmax(counts)])
    if np.max(counts) <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    markers = [int(v) for v, n in zip(colors, counts) if n == 2]
    if len(markers) != 1:
        raise ValueError("terminal role must occur exactly twice")
    terminal = markers[0]
    wall = next(int(v) for v in colors if v not in (background, terminal))
    centers = tuple(tuple(map(int, p)) for p in np.argwhere(a == terminal))
    gates = []
    chamber_mask = np.zeros_like(a, dtype=bool)
    for r, c in centers:
        if not (1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1):
            raise ValueError("terminal chamber outside canvas")
        area = np.zeros_like(chamber_mask)
        area[r-1:r+2, c-1:c+2] = True
        if (area & chamber_mask).any():
            raise ValueError("overlapping chambers")
        chamber_mask |= area
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if a[r+dr, c+dc] != wall:
                raise ValueError("chamber corner must be a wall")
        exits = []
        for k, (dr, dc) in enumerate(STEPS):
            if a[r+dr, c+dc] == background:
                exits.append(k)
            elif a[r+dr, c+dc] != wall:
                raise ValueError("invalid chamber side")
        if len(exits) != 1:
            raise ValueError("chamber needs exactly one gate")
        k = exits[0]
        dr, dc = STEPS[k]
        if not (0 <= r+2*dr < a.shape[0] and 0 <= c+2*dc < a.shape[1]):
            raise ValueError("gate trapped at canvas edge")
        gates.append(k)
    return a, background, wall, terminal, centers, tuple(gates), chamber_mask


def _distances(blocked, start):
    h, w = blocked.shape
    result = np.full((h, w), -1, dtype=np.int16)
    result[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        distance = int(result[r, c]) + 1
        for dr, dc in STEPS:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and result[rr, cc] < 0:
                result[rr, cc] = distance
                queue.append((rr, cc))
    return result


def _route_union(blocked, centers):
    a = _distances(blocked, centers[0])
    length = int(a[centers[1]])
    if length < 0:
        raise ValueError("terminals disconnected after repair")
    b = _distances(blocked, centers[1])
    return (a >= 0) & (b >= 0) & (a+b == length), length, a


def _interpret(value):
    a, bg, wall, marker, centers, gates, chamber = _read_input(value)
    clean, added, erased, evidence = _repair_scaffold(a == wall)
    if np.any(clean[chamber] != (a == wall)[chamber]):
        raise ValueError("cleanup cannot alter a gated chamber")
    union, length, distances = _route_union(clean, centers)
    route = union & (a != marker) & ~clean
    return dict(a=a, bg=bg, wall=wall, marker=marker, centers=centers,
                gates=gates, chamber=chamber, clean=clean, added=added,
                erased=erased, evidence=evidence, union=union, route=route,
                length=length, distances=distances)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every edit from this input, including the cleaned obstacle map."""
    info = _interpret(input_grid)
    out = info['a'].copy()
    out[info['erased']] = info['bg']
    out[info['added']] = info['wall']
    out[info['route']] = ROUTE
    return out


def _chamber(mask, center, gate):
    r, c = center
    mask[r-1:r+2, c-1:c+2] = True
    mask[r, c] = False
    dr, dc = STEPS[gate]
    mask[r+dr, c+dc] = False


def _thick(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).any())


def _diagonal_strokes(mask):
    """Erased fragments must be separate straight diagonal pairs or triples."""
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    while cells:
        start = min(cells)
        cells.remove(start)
        component, stack = [], [start]
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nxt = r+dr, c+dc
                    if (dr or dc) and nxt in cells:
                        cells.remove(nxt)
                        stack.append(nxt)
        component.sort()
        if len(component) not in (2, 3):
            return False
        offsets = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(component, component[1:])]
        if len(set(offsets)) != 1 or offsets[0] not in ((1, 1), (1, -1)):
            return False
    return True


def _scene(rng, rich):
    """Grow a variable forest of blocking trunks, arms, and occasional loops."""
    h, w = int(rng.integers(18, 31)), int(rng.integers(22, 31))
    wall = np.zeros((h, w), dtype=bool)
    rows = tuple(int(rng.integers(5, h-5)) for _ in range(2))
    centers = ((rows[0], int(rng.integers(1, 4))),
               (rows[1], w-int(rng.integers(2, 5))))
    gates = tuple(map(int, rng.integers(0, 4, size=2)))
    for center, gate in zip(centers, gates):
        # Side gates at an outer boundary need one more cell outside the shell.
        r, c = center
        dr, dc = STEPS[gate]
        if not (0 <= r+2*dr < h and 0 <= c+2*dc < w):
            return None
        _chamber(wall, center, gate)
    protected = np.zeros_like(wall)
    for r, c in centers:
        protected[max(0, r-3):min(h, r+4), max(0, c-3):min(w, c+4)] = True
    columns = list(range(7, w-7))
    rng.shuffle(columns)
    trunks = []
    count = int(rng.integers(1, 2+rich))
    for c in columns:
        if any(abs(c-other[0]) < 4 for other in trunks):
            continue
        top = int(rng.integers(1, 4))
        bottom = int(rng.integers(h-4, h-1))
        wall[top:bottom+1, c] = True
        trunks.append((c, top, bottom))
        if len(trunks) == count:
            break
    if not trunks:
        return None
    for c, top, bottom in trunks:
        levels = list(range(top, bottom+1))
        rng.shuffle(levels)
        chosen = []
        for row in levels:
            if any(abs(row-other) < 4 for other in chosen):
                continue
            direction = int(rng.choice([-1, 1]))
            length = int(rng.integers(2, 6))
            end = int(np.clip(c+direction*length, 5, w-6))
            trial = wall.copy()
            trial[row, min(c, end):max(c, end)+1] = True
            if not _thick(trial) and not np.any((trial & ~wall) & protected):
                wall = trial
                chosen.append(row)
            if len(chosen) >= int(rng.integers(1, 3+rich)):
                break
    # A true three-cell opening contrasts with supported repairs. It may lie in
    # an arm or a trunk; its lack of a matching spur is the visible distinction.
    possibilities = _damage_options(wall, protected)
    if not possibilities:
        return None
    gap, _ = possibilities[int(rng.integers(len(possibilities)))]
    for cell in gap:
        wall[cell] = False
    if not np.array_equal(_repair_scaffold(wall)[0], wall):
        return None
    try:
        union, length, _ = _route_union(wall, centers)
    except ValueError:
        return None
    if int(union.sum()) <= length + 1:
        return None
    return wall, centers, gates, protected, tuple(gap)


def _damage_options(wall, protected):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(wall):
            r, c = int(r), int(c)
            if not all(_at(wall, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(wall, r-dr, c-dc) or _at(wall, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(protected[cell] for cell in gap):
                continue
            if any(_at(wall, rr+dc, cc+dr) or _at(wall, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    spur = tuple((ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2))
                    if all(0 <= rr < wall.shape[0] and 0 <= cc < wall.shape[1]
                           and not wall[rr, cc] and not protected[rr, cc] for rr, cc in spur):
                        result.append((gap, spur))
    return result


def _corrupt(rng, clean, centers, protected, target, axis=None):
    inp = clean.copy()
    union, _, _ = _route_union(clean, centers)
    options = _damage_options(clean, protected)
    accepted = 0
    touched = set()
    for index in rng.permutation(len(options)):
        gap, spur = options[int(index)]
        if axis is not None and (gap[0][0] == gap[-1][0]) != (axis == 'h'):
            continue
        cells = gap + spur
        if any(any(abs(r-rr) <= 1 and abs(c-cc) <= 1 for rr, cc in touched) for r, c in cells):
            continue
        trial = inp.copy()
        for cell in gap:
            trial[cell] = False
        for cell in spur:
            trial[cell] = True
        try:
            recovered, added, _, evidence = _repair_scaffold(trial)
        except ValueError:
            continue
        if not np.array_equal(recovered, clean) or len(evidence) != accepted+1:
            continue
        if accepted == 0:
            # Necessity concerns route cells, not merely visible wall edits.
            without_repair, _, _ = _route_union(clean & ~added, centers)
            if np.array_equal(without_repair, union):
                continue
        inp = trial
        touched.update(cells)
        accepted += 1
        if accepted == target:
            return inp
    return None


def _add_decoys(rng, inp, clean, centers, protected, richness):
    """Place a deletable diagonal on a real route, plus optional off-route ones."""
    route, _, _ = _route_union(clean, centers)
    places = np.argwhere(route & ~protected & ~inp)
    for index in rng.permutation(len(places)):
        r, c = map(int, places[int(index)])
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            cells = [(r, c), (r+dr, c+dc)]
            if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                       and not inp[rr, cc] and not protected[rr, cc] for rr, cc in cells):
                continue
            trial = inp.copy()
            for cell in cells:
                trial[cell] = True
            try:
                if (np.array_equal(_repair_scaffold(trial)[0], clean)
                        and _diagonal_strokes(trial & ~clean)):
                    inp = trial
                    break
            except ValueError:
                pass
        else:
            continue
        break
    else:
        return None
    for _ in range(int(rng.integers(richness+2))):
        for _ in range(12):
            r, c = int(rng.integers(inp.shape[0])), int(rng.integers(inp.shape[1]))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(r+k*dr, c+k*dc) for k in range(int(rng.choice([2, 3])))]
            if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                       and not inp[rr, cc] and not clean[rr, cc] and not protected[rr, cc] for rr, cc in cells):
                continue
            trial = inp.copy()
            for cell in cells:
                trial[cell] = True
            try:
                if (np.array_equal(_repair_scaffold(trial)[0], clean)
                        and _diagonal_strokes(trial & ~clean)):
                    inp = trial
                    break
            except ValueError:
                pass
    return inp


def _make_pair(rng, target, richness, turns, axis=None):
    for _ in range(80):
        scene = _scene(rng, richness)
        if scene is None:
            continue
        clean, centers, gates, protected, intentional = scene
        inp = _corrupt(rng, clean, centers, protected, target, axis)
        if inp is None:
            continue
        inp = _add_decoys(rng, inp, clean, centers, protected, richness)
        if inp is None:
            continue
        palette = tuple(map(int, rng.choice([c for c in range(10) if c != ROUTE], 3, replace=False)))
        bg, wall, marker = palette
        grid = np.where(inp, wall, bg).astype(np.int8)
        expected = np.where(clean, wall, bg).astype(np.int8)
        union, _, _ = _route_union(clean, centers)
        expected[union & ~clean] = ROUTE
        for center in centers:
            grid[center] = expected[center] = marker
        if rng.integers(2):
            grid, expected = grid[:, ::-1], expected[:, ::-1]
        grid, expected = np.rot90(grid, turns).copy(), np.rot90(expected, turns).copy()
        # Compare clean construction with input-only recovery over the full scene.
        if not np.array_equal(solve(grid), expected):
            continue
        return {'input': grid, 'output': expected}
    raise RuntimeError("bounded sampler could not find a recoverable maze")


def _counts(num_train, num_test):
    for name, v, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(4))
    shift = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            # Training includes both axes even at 2/1 counts. Test composes two
            # independent repairs, and in one regime adds more blocking trunks.
            target = 1 if split == 'train' else 2
            richness = 1 if split == 'train' else 1+shift
            turns = (offset+i) % 4 if i < 2 and split == 'train' else int(rng.integers(4))
            for _ in range(8):
                pair = _make_pair(rng, target, richness, turns,
                                  axis='v' if split == 'train' and i < 2 else None)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
    info = _interpret(pair['input'])
    out = _grid(pair['output'])
    a = info['a']
    _require(a.shape == out.shape and np.array_equal(solve(a), out), "incorrect full transformation")
    _require(info['evidence'] and info['erased'].any(), "repair and deletion must both occur")
    _require(_diagonal_strokes(info['erased']), "erased clutter must be separate diagonal pairs/triples")
    _require(not _thick(info['clean']), "wall thickness must remain one")
    _require(int(info['union'].sum()) > info['length']+1, "need multiple equal shortest routes")
    _require(np.any((a == info['bg']) & (out == info['bg'])), "off-route background must remain")
    _require(np.all(out[a == info['marker']] == info['marker']), "terminal changed")
    _require(np.all(out[info['clean']] == info['wall']), "wall changed")
    _require(np.all(out[info['route']] == ROUTE), "missing route cell")
    for (r, c), gate in zip(info['centers'], info['gates']):
        dr, dc = STEPS[gate]
        _require(out[r+dr, c+dc] == ROUTE, "mandatory gate not colored")
    # Strong causal ablations: remove just repair, or just diagonal erasure.
    no_repair, _, _ = _route_union(info['clean'] & ~info['added'], info['centers'])
    _require(not np.array_equal(no_repair, info['union']), "repair is irrelevant to routing")
    _require(np.any(info['erased'] & info['route']), "deletion is irrelevant to routing")
    unsupported = [gap for gap in _gaps(info['clean'])
                   if (_at(info['clean'], gap[0]-gap[2], gap[1]-gap[3])
                       or _at(info['clean'], gap[0]+5*gap[2], gap[1]+5*gap[3]))]
    _require(unsupported, "need a preserved three-cell opening")
    info['unsupported'] = unsupported
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split type")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    training_axes = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            if split == 'train':
                training_axes.update('h' if d[2] == 0 else 'v' for d, _ in info['evidence'])
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
    _require(training_axes == {'h', 'v'}, "training must expose both repair axes")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for s in ('train', 'test'):
        for pair in puzzle[s]:
            for k in ('input', 'output'):
                a = np.asarray(pair[k], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, retaining only hashes/metrics; never save bulk payloads."""
    shapes, palettes, geometries, lengths, sizes, axes, gates, fingerprints = (set() for _ in range(8))
    pairs = repaired = erased_on_route = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                pairs += 1
                repaired += len(info['evidence'])
                erased_on_route += int(np.count_nonzero(info['erased'] & info['route']))
                shapes.add(info['a'].shape)
                palettes.add((info['bg'], info['wall'], info['marker']))
                geometries.add(hashlib.sha256(info['clean'].tobytes()).hexdigest())
                lengths.add(info['length'])
                sizes.add(int(info['route'].sum()))
                axes.update('h' if d[2] == 0 else 'v' for d, _ in info['evidence'])
                gates.add(info['gates'])
        # At least 2 supported repairs is the held-out compositional shift.
        _require(all(len(_interpret(p['input'])['evidence']) >= 2 for p in puzzle['test']), "missing test shift")
    _require(len(fingerprints) == 42 and len(geometries) >= 170, "structural diversity")
    _require(len(shapes) >= 45 and len(palettes) >= 90, "geometry/palette diversity")
    _require(len(lengths) >= 20 and len(sizes) >= 35 and axes == {'h', 'v'}, "route/repair diversity")
    _require(len(gates) >= 12, "gate diversity")
    sample = generate_puzzle(0, 2, 1)
    _require(_fingerprint(sample) == _fingerprint(generate_puzzle(0, 2, 1)), "nondeterminism")
    for pair in sample['train'] + sample['test']:
        before, after = pair['input'], pair['output']
        for turns in range(4):
            for flip in (False, True):
                a, b = np.rot90(before, turns), np.rot90(after, turns)
                if flip:
                    a, b = a[:, ::-1], b[:, ::-1]
                _require(np.array_equal(solve(a), b), "rotation/reflection covariance")
    # Mutations exercise failures that equality of schema alone would miss.
    for kind in ('omit_repair', 'omit_route', 'fill_unsupported'):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in ('train', 'test')}
        pair = damaged['train'][0]
        info = _interpret(pair['input'])
        if kind == 'omit_repair':
            cell = tuple(np.argwhere(info['added'])[0])
            pair['output'][cell] = info['bg']
        elif kind == 'omit_route':
            cell = tuple(np.argwhere(info['route'])[0])
            pair['output'][cell] = info['bg']
        else:
            r, c, dr, dc = _gaps(info['clean'])[0]
            pair['output'][r+dr, c+dc] = info['wall']
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {kind}")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            puzzle = generate_puzzle(19, n, t)
            _require((len(puzzle['train']), len(puzzle['test'])) == (n, t), "explicit counts")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    return dict(seeds=42, pairs=pairs, puzzles=len(fingerprints), shapes=len(shapes),
                palettes=len(palettes), wall_geometries=len(geometries),
                route_lengths=len(lengths), route_sizes=len(sizes),
                gate_pairs=len(gates), repairs=repaired, erased_route_cells=erased_on_route)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
