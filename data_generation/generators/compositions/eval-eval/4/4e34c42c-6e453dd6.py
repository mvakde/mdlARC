"""Laminated window projectors: registration, enclosure, and overlap depth.

Only stdlib and NumPy. No evidence files, learned state, or disk writes are used.
See task_spec.md for the rule, including the visible rail-ink signal convention.
"""
from __future__ import annotations

from collections import deque
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _holes(mask):
    h, w = mask.shape
    return [g for g in _groups(zip(*np.nonzero(~mask)))
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in g)]


def _scene(grid):
    """Rotate coordinates for calculation; individual fragments never rotate."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(1, w - 1):
            ink = int(a[0, col])
            if not np.all(a[:, col] == ink) or np.count_nonzero(a == ink) != h:
                continue
            bg = int(a[0, -1])
            if ink == bg or not np.all(a[:, col + 1:] == bg):
                continue
            work = a[:, :col]
            colors, counts = np.unique(work[work != bg], return_counts=True)
            if len(colors) < 3 or np.count_nonzero(counts == counts.max()) != 1:
                continue
            body = int(colors[counts.argmax()])
            candidates.append((a, turn, bg, ink, body, col))
    _require(len(candidates) == 1, "need one exclusive rail and one blank projection side")
    return candidates[0]


def _glyphs(part, body):
    """A rivet is a whole connected cluster of non-body foreground ink."""
    result = []
    for group in _groups(p for p, v in part.items() if v != body):
        r0 = min(r for r, c in group)
        c0 = min(c for r, c in group)
        signature = tuple(sorted((r - r0, c - c0, part[r, c]) for r, c in group))
        _require(len(group) >= 2 and len({part[p] for p in group}) >= 2,
                 "a rivet needs at least two pixels and two ink colors")
        result.append((signature, (r0, c0)))
    _require(bool(result), "every fragment needs a rivet")
    _require(len({s for s, p in result}) == len(result), "rivet repeated within one fragment")
    return result


def _project(canonical, bg, ink, col, union, depth):
    """Only the assembled material, without the rail, encloses cavities."""
    mask = np.zeros((canonical.shape[0], col), dtype=bool)
    for p in union:
        mask[p] = True
    holes = _holes(mask)
    extreme = max(c for r, c in union)
    _require(extreme == col - 1, "assembled frame must be docked at the rail")
    contact = {r for hole in holes for r, c in hole if c == extreme - 1}
    rays = {r for r in contact if mask[r, extreme] and depth[r, extreme] >= 2}
    out = np.full(canonical.shape, bg, dtype=np.int8)
    out[:, col] = ink
    for p, v in union.items():
        out[p] = v
    for r in rays:
        out[r, col + 1:] = ink
    return out, mask, holes, contact, rays


def _analyze(input_grid):
    a, turn, bg, ink, body, col = _scene(_grid(input_grid))
    parts = [{p: int(a[p]) for p in g}
             for g in _groups(zip(*np.nonzero(a[:, :col] != bg)))]
    _require(len(parts) >= 2, "need a mounted frame and loose fragments")
    anchors = [i for i, p in enumerate(parts) if any(c == col - 1 for r, c in p)]
    _require(len(anchors) == 1, "exactly one fragment is initially docked")
    occurrences = {}
    for i, part in enumerate(parts):
        for signature, pos in _glyphs(part, body):
            occurrences.setdefault(signature, []).append((i, pos))
    adjacency = [[] for _ in parts]
    for copies in occurrences.values():
        _require(len(copies) >= 2, "unmatched rivet")
        i, (r, c) = copies[0]
        for j, (u, v) in copies[1:]:
            delta = (r - u, c - v)
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {anchors[0]: (0, 0)}
    queue = deque([anchors[0]])
    while queue:
        i = queue.popleft()
        dr, dc = offsets[i]
        for j, (r, c) in adjacency[i]:
            proposed = dr + r, dc + c
            if j in offsets:
                _require(offsets[j] == proposed, "inconsistent registration cycle")
            else:
                offsets[j] = proposed
                queue.append(j)
    _require(len(offsets) == len(parts), "disconnected registration graph")
    union = {}
    depth = np.zeros((a.shape[0], col), dtype=np.int16)
    placed = []
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        view = {}
        for (r, c), v in part.items():
            p = (r + dr, c + dc)
            _require(0 <= p[0] < a.shape[0] and 0 <= p[1] < col,
                     "registered material crosses a boundary")
            _require(p not in union or union[p] == v, "unequal-color overlap")
            union[p] = view[p] = v
            depth[p] += 1
        placed.append(view)
    _require(len(_groups(union)) == 1, "the assembled frame must be connected")
    out, mask, holes, contact, rays = _project(a, bg, ink, col, union, depth)
    enclosed = set().union(*holes) if holes else set()
    edge = col - 1
    open_double = {r for r in range(a.shape[0])
                   if not mask[r, edge - 1] and (r, edge - 1) not in enclosed
                   and mask[r, edge] and depth[r, edge] >= 2}
    recessed_double = {r for hole in holes for r, c in hole
                       if c + 1 < edge and mask[r, c + 1] and depth[r, c + 1] >= 2}
    # Source fragments have no holes in the sampled subset. The rule and solver
    # also accept fragments already containing holes; enclosure is always final.
    original_holes = 0
    for part in parts:
        rr, cc = zip(*part)
        r0, c0 = min(rr), min(cc)
        local = np.zeros((max(rr) - r0 + 3, max(cc) - c0 + 3), dtype=bool)
        for r, c in part:
            local[r - r0 + 1, c - c0 + 1] = True
        original_holes += len(_holes(local))
    ur, uc = min(r for r, c in union), min(c for r, c in union)
    geometry = tuple(sorted((r - ur, c - uc) for r, c in union))
    info = dict(shape=tuple(a.shape), turn=turn, background=bg, body=body, rail=ink,
                fragments=len(parts), keys=len(occurrences), holes=len(holes),
                active_holes=sum(any(c == edge - 1 for r, c in hole) for hole in holes),
                recessed_holes=sum(all(c != edge - 1 for r, c in hole) for hole in holes),
                ray_rows=tuple(sorted(rays)), open_double_rows=tuple(sorted(open_double)),
                recessed_double_rows=tuple(sorted(recessed_double)),
                single_contact_rows=tuple(sorted(contact - rays)),
                contact_rows=tuple(sorted(contact)), original_holes=original_holes,
                max_depth=int(depth.max()), band=a.shape[1] - col - 1,
                glyph_shapes=tuple(sorted({(max(r for r, c, v in s) + 1,
                                            max(c for r, c, v in s) + 1) for s in occurrences})),
                geometry_hash=hashlib.sha256(repr(geometry).encode()).hexdigest(),
                moves=tuple(offsets[i] for i in range(len(parts))))
    return np.rot90(out, -turn).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from pixels, including all palette and registration roles."""
    return _analyze(input_grid)[0]


def _normalize(part):
    r0, c0 = min(r for r, c in part), min(c for r, c in part)
    return {(r - r0, c - c0): v for (r, c), v in part.items()}


def _make_layers(rng, states, colors, force_partial=False):
    """Construct one frame, then genuine overlapping views of its material."""
    bg, ink, body, *key_colors = colors
    n = len(states)
    width = int(rng.integers(8, 10 if n == 4 else 12))
    heights = [int(rng.integers(3, 5 if n == 4 else 7)) for _ in states]
    if force_partial:
        heights[states.index('active')] = int(rng.integers(4, 7))
    gaps = list(map(int, rng.integers(1, 3, size=n - 1)))
    height = 2 + sum(heights) + sum(gaps)
    _require(height <= 30, "frame exceeds canvas")
    root = {(r, 0): body for r in range(height)}
    root.update({(0, c): body for c in range(width)})  # fixed docking handle
    layers, used = [], set()
    top = 2
    for i, (state, h) in enumerate(zip(states, heights)):
        bottom = top + h - 1
        front = width - 1 - (int(rng.integers(1, 3)) if state == 'decoy' else 0)
        back = int(rng.integers(1, 3))
        key_start = back
        key_len = int(rng.integers(2, min(4, front - key_start - 1) + 1))
        from_top = bool(rng.integers(2))
        bar = top if from_top else bottom
        other = bottom if from_top else top
        step = 1 if from_top else -1
        # Nonperiodic, visibly colored rivets; identities are local to this pair.
        for _ in range(100):
            word = list(map(int, rng.choice(key_colors, key_len)))
            if len(set(word)) < 2:
                continue
            glyph = {(bar, key_start + c): v for c, v in enumerate(word)}
            if rng.random() < .55:
                c = int(rng.integers(key_len))
                glyph[bar + step, key_start + c] = int(rng.choice(key_colors))
            signature = tuple(sorted((r, c, v) for (r, c), v in _normalize(glyph).items()))
            if signature not in used:
                used.add(signature)
                break
        else:
            raise AssertionError("could not sample distinct rivets")
        # Left wall can have a stepped back edge; it never reaches the front.
        for r in range(top, bottom + 1):
            thickness = int(rng.integers(1, back + 1))
            root.update({(r, c): body for c in range(thickness)})
        root.update({(bar, c): body for c in range(front + 1)})
        stop = int(rng.integers(back + 1, front)) if state == 'open' else front + 1
        root.update({(other, c): body for c in range(stop)})
        root.update(glyph)
        main = {(bar, c): body for c in range(key_start, front + 1)}
        main.update({(r, front): body for r in range(top, bottom + 1)})
        # A second prong can partition one chamber into side-by-side cavities.
        # It is supplied by the loose piece, so the root still has no holes.
        if front >= key_start + key_len + 3 and rng.random() < .4:
            separator = int(rng.integers(key_start + key_len, front - 1))
            main.update({(r, separator): body for r in range(top, bottom + 1)})
        main.update(glyph)
        layers.append(main)
        if state != 'single':
            limit = h - 3 if force_partial and state == 'active' else h - 1
            length = int(rng.integers(1, limit + 1))
            sleeve = {(bar, c): body for c in range(key_start, front + 1)}
            sleeve.update({(bar + step * d, front): body for d in range(length + 1)})
            sleeve.update(glyph)
            layers.append(sleeve)
        top = bottom + 1 + (gaps[i] if i < n - 1 else 0)
    return [root] + layers, height, width


def _pack(rng, layers, frame_h, frame_w, bg, ink):
    parts = [_normalize(p) for p in layers]
    shapes = [(max(r for r, c in p) + 1, max(c for r, c in p) + 1) for p in parts]
    for attempt in range(18):
        height = int(rng.integers(max(frame_h, 18), 31)) if attempt < 12 else 30
        low = max(frame_w + 10, 2 * frame_w + 1)
        col = int(rng.integers(low, 29)) if attempt < 12 else 28
        band = int(rng.integers(1, min(6, 29 - col) + 1))
        a = np.full((height, col + 1 + band), bg, dtype=np.int8)
        a[:, col] = ink
        blocked = np.zeros((height, col), dtype=bool)
        root_r, root_c = int(rng.integers(height - frame_h + 1)), col - frame_w

        def place(part, dr, dc):
            for (r, c), v in part.items():
                r, c = r + dr, c + dc
                a[r, c] = v
                blocked[max(0, r - 1):min(height, r + 2), max(0, c - 1):min(col, c + 2)] = True

        place(parts[0], root_r, root_c)
        order = sorted(range(1, len(parts)), key=lambda j: (-len(parts[j]), float(rng.random())))
        for j in order:
            ph, pw = shapes[j]
            nr, nc = height - ph + 1, col - pw  # loose pieces may not touch rail
            if nr <= 0 or nc <= 0:
                break
            feasible = np.ones((nr, nc), dtype=bool)
            for r, c in parts[j]:
                feasible &= ~blocked[r:r + nr, c:c + nc]
            choices = np.argwhere(feasible)
            if not len(choices):
                break
            dr, dc = map(int, choices[int(rng.integers(len(choices)))])
            place(parts[j], dr, dc)
        else:
            union = {}
            depth = np.zeros((height, col), dtype=np.int16)
            for layer in layers:
                for (r, c), v in layer.items():
                    p = r + root_r, c + root_c
                    _require(p not in union or union[p] == v, "builder color conflict")
                    union[p] = v
                    depth[p] += 1
            out = _project(a, bg, ink, col, union, depth)[0]
            return a, out
    raise AssertionError("bounded fragment packing exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, int(rng.integers(5, 8)), replace=False)))
    vary_palette = bool(rng.integers(2))
    turn = int(rng.integers(4))
    first = [['active', 'open'], ['single', 'decoy']]
    if rng.integers(2):
        first.reverse()
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index < 2:
                states = first[index].copy()
            else:
                n = int(rng.integers(3, 5)) if split == 'test' else int(rng.integers(1, 3))
                states = list(map(str, rng.choice(['active', 'open', 'single', 'decoy'], n)))
                # Queries combine the demonstrated distinctions, including a
                # positive window, without a new operation or color convention.
                if split == 'test':
                    states[0] = 'active'
            rng.shuffle(states)
            colors = (list(map(int, rng.choice(10, len(palette), replace=False)))
                      if vary_palette else palette)
            for _ in range(12):
                try:
                    layers, fh, fw = _make_layers(rng, states, colors,
                                                  force_partial=split == 'train' and index < 2 and 'active' in states)
                    inp, out = _pack(rng, layers, fh, fw, colors[0], colors[1])
                    inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
                    key = _signature(inp)
                    if key in seen:
                        continue
                    predicted, info = _analyze(inp)
                    _require(info['body'] == colors[2], "ambiguous body frequency")
                    _require(np.array_equal(predicted, out), "registration differs from construction")
                    _require(info['original_holes'] == 0, "unintended source cavity")
                    seen.add(key)
                    result[split].append({'input': inp, 'output': out})
                    break
                except AssertionError:
                    continue
            else:
                raise AssertionError("bounded pair generation exhausted")
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected, info = _analyze(inp)
            _require(inp.shape == out.shape and np.array_equal(out, expected), "wrong assembly or rays")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "unencoded output ink")
            if split == 'train':
                train_info.append(info)
    # Pixel-derived curriculum checks, separate from the solver's valid domain.
    _require(any(i['ray_rows'] for i in train_info), "no positive demonstration")
    _require(any(set(i['open_double_rows']) - set(i['ray_rows']) for i in train_info),
             "enclosure is not distinguished from a double open wall")
    _require(any(set(i['recessed_double_rows']) - set(i['ray_rows']) for i in train_info),
             "extreme-front selection is not demonstrated")
    _require(any(i['single_contact_rows'] for i in train_info), "single/double distinction is missing")
    _require(any(i['ray_rows'] and i['single_contact_rows'] for i in train_info),
             "training must resolve partial reinforcement row by row")
    _require(any(i['holes'] > i['original_holes'] and i['ray_rows'] for i in train_info),
             "training lacks cavities created by cooperation")


def self_test():
    """42 streamed seeds, meaningful counterfactuals, counts, and corruption.

    No files, process workers, retained puzzle corpus, or execution traces.
    Only small sets of hashes/statistics survive each puzzle iteration.
    """
    hashes, shapes, silhouettes, glyph_shapes = set(), set(), set(), set()
    backgrounds, bodies, rails, directions, counts, depths, bands = (set() for _ in range(7))
    total = partial = open_cases = recessed = silent = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, i = _analyze(pair['input'])
                _require(np.array_equal(expected, pair['output']), "label mismatch")
                for a in pair.values():
                    fingerprint.update(repr(a.shape).encode())
                    fingerprint.update(a.tobytes())
                shapes.add(i['shape'])
                silhouettes.add(i['geometry_hash'])
                glyph_shapes.update(i['glyph_shapes'])
                backgrounds.add(i['background'])
                bodies.add(i['body'])
                rails.add(i['rail'])
                directions.add(i['turn'])
                counts.add(i['fragments'])
                depths.add(i['max_depth'])
                bands.add(i['band'])
                total += 1
                partial += bool(i['ray_rows'] and i['single_contact_rows'])
                open_cases += bool(set(i['open_double_rows']) - set(i['ray_rows']))
                recessed += bool(set(i['recessed_double_rows']) - set(i['ray_rows']))
                silent += not bool(i['ray_rows'])
        hashes.add(fingerprint.hexdigest())
    _require(len(hashes) == 42, "seed diversity collapsed")
    _require(len(shapes) >= 30 and len(silhouettes) >= 150, "geometry diversity collapsed")
    _require(len(counts) >= 5 and len(glyph_shapes) >= 5, "part or rivet diversity collapsed")
    _require(directions == set(range(4)), "missing rail direction")
    _require(backgrounds == bodies == rails == set(range(10)), "palette roles collapsed")
    _require(partial >= 42 and open_cases >= 42 and recessed >= 42 and silent >= 42,
             "counterexamples collapsed")
    _require(1 in bands and len(bands) >= 3 and max(depths) >= 3, "layer/band diversity collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(814, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "requested counts ignored")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(117, 2, 1), generate_puzzle(117, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a),
             "seed is not deterministic")
    # Whole-scene rotations and arbitrary palette permutations must commute
    # with an input-only solve; neither the source placement nor color names
    # are allowed to supply hidden coordinates or signal ink.
    pair = p['train'][0]
    lut = np.array([7, 0, 8, 3, 9, 2, 4, 1, 5, 6], dtype=np.int8)
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(lut[pair['input']], turn)),
                                np.rot90(lut[pair['output']], turn)), "equivariance failed")
    _regression_test()
    q['test'][0]['output'][0, 0] = (int(q['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=total, puzzles=len(hashes), canonical_shapes=len(shapes),
                silhouettes=len(silhouettes), fragment_counts=sorted(counts),
                glyph_shapes=sorted(glyph_shapes), partial_pairs=partial,
                open_counterexamples=open_cases, recessed_counterexamples=recessed,
                silent_pairs=silent, max_depths=sorted(depths), bands=sorted(bands))


def _regression_test():
    """Small designed contrasts with literal expected ray rows, not RNG labels."""
    root = {(r, 0): 1 for r in range(7)}
    root.update({(r, c): 1 for r in (0, 2, 6) for c in range(8)})
    root[2, 2], root[2, 3] = 2, 3
    main = {(2, c): root[2, c] for c in range(2, 8)}
    main.update({(r, 7): 1 for r in range(2, 7)})
    sleeve = {(2, c): root[2, c] for c in range(2, 8)}
    sleeve.update({(r, 7): 1 for r in range(2, 5)})

    def scene(base, pieces):
        a = np.zeros((16, 24), dtype=np.int8)
        a[:, 20] = 8
        mount_c = 19 - max(c for r, c in base)
        for (r, c), v in base.items():
            a[r + 1, c + mount_c] = v
        slots = [(1, 1), (10, 1), (10, 10)]
        for piece, (dr, dc) in zip(pieces, slots):
            for (r, c), v in _normalize(piece).items():
                a[r + dr, c + dc] = v
        return a

    inp = scene(root, [main, sleeve])
    out, info = _analyze(inp)
    _require(info['ray_rows'] == (4, 5), "partial reinforced cavity regression")
    _require(np.all(out[[4, 5], 21:] == 8) and np.all(out[6, 21:] == 0),
             "literal ray endpoints regression")
    # Removing a geometrically redundant fragment preserves the assembled
    # work area exactly, yet removes every ray. A union-only pipeline fails.
    single, single_info = _analyze(scene(root, [main]))
    _require(np.array_equal(single[:, :21], out[:, :21]) and not single_info['ray_rows'],
             "overlap multiplicity was discarded")
    triple, triple_info = _analyze(scene(root, [main, sleeve, sleeve]))
    _require(np.array_equal(triple, out) and triple_info['max_depth'] >= 3,
             "at-least-two must not become exactly-two or parity")
    opened = {p: v for p, v in root.items() if not (p[0] == 6 and p[1] >= 2)}
    _, open_info = _analyze(scene(opened, [main, sleeve]))
    _require(not open_info['ray_rows'] and open_info['open_double_rows'], "enclosure regression")
    extended = dict(root)
    extended[0, 8] = 1
    _, decoy_info = _analyze(scene(extended, [main, sleeve]))
    _require(not decoy_info['ray_rows'] and decoy_info['recessed_double_rows'], "global extreme regression")
    # Chain registration: the small wall strip shares a vertical rivet only
    # with the main piece, which shares a horizontal rivet with the root.
    bridge = dict(main)
    bridge[3, 7], bridge[4, 7] = 2, 3
    strip = {(3, 7): 2, (4, 7): 3, (5, 7): 1}
    _, chain_info = _analyze(scene(root, [bridge, strip]))
    _require(chain_info['ray_rows'] == (4, 5, 6), "multi-hop registration regression")
    # An unpaired glyph has no interpretable pose and must not be guessed.
    bad = inp.copy()
    bad[10, 1] = 4
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("unmatched rivet accepted")
