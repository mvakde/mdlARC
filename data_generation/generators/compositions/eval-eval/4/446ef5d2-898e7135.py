"""Socket-and-seam mosaics: a joint geometric and colored jigsaw.

The solver reads only the supplied input. Generator labels are drawn directly
from a logical mosaic, then checked against the complete input-only search.
"""
from __future__ import annotations

import math
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        for r, c in stack:
            for dr, dc in _N4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(stack)
    return result


def _bbox(cells):
    q = np.asarray(cells)
    r, c = q.min(axis=0)
    y, x = q.max(axis=0) + 1
    return int(r), int(y), int(c), int(x)


def _orientations(tile):
    seen, result = set(), []
    for base in (tile, tile[:, ::-1]):
        for k in range(4):
            a = np.rot90(base, k).copy()
            key = (a.shape, a.tobytes())
            if key not in seen:
                seen.add(key)
                result.append(a)
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _decode(g, cells, scale, background):
    r, y, c, x = _bbox(cells)
    h, w = y - r, x - c
    if h % scale or w % scale:
        return None
    crop = g[r:y, c:x]
    logical = crop[::scale, ::scale].copy()
    if not np.array_equal(crop, np.repeat(np.repeat(logical, scale, 0), scale, 1)):
        return None
    logical[logical == background] = -1
    if int(np.count_nonzero(logical >= 0)) * scale * scale != len(cells):
        return None
    if len(_components(logical >= 0)) != 1:
        return None
    return logical


def _interpretations(g):
    """Find board and scale geometrically, without looking at an output."""
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= g.size // 2:
        raise ValueError("the background must occupy more than half the canvas")
    components = _components(g != bg)
    for component in components:
        r, y, c, x = _bbox(component)
        board = g[r:y, c:x].copy()
        if min(board.shape) < 5 or max(board.shape) > 15:
            continue
        carrier = int(board[0, 0])
        if (carrier == bg or not np.all(board[0] == carrier)
                or not np.all(board[-1] == carrier)
                or not np.all(board[:, 0] == carrier)
                or not np.all(board[:, -1] == carrier)):
            continue
        holes = board == bg
        if not holes.any():
            continue
        ports = (board != bg) & (board != carrier)
        if not ports.any():
            continue
        if any(sum(0 <= rr + dr < board.shape[0] and 0 <= cc + dc < board.shape[1]
                       and holes[rr + dr, cc + dc] for dr, dc in _N4) != 1
               for rr, cc in np.argwhere(ports)):
            continue
        outside = np.ones(g.shape, dtype=bool)
        outside[r:y, c:x] = False
        # Board and loose objects have an empty one-cell moat.
        moat = np.zeros(g.shape, dtype=bool)
        moat[max(0, r-1):min(g.shape[0], y+1), max(0, c-1):min(g.shape[1], x+1)] = True
        if np.any(moat & outside & (g != bg)):
            continue
        external = _components(outside & (g != bg))
        large = [p for p in external if len(p) >= 4]
        noise = [p for p in external if len(p) < 4]
        if not 2 <= len(large) <= 8:
            continue
        area = sum(map(len, large))
        hole_area = int(holes.sum())
        quotient, remainder = divmod(area, hole_area)
        scale = math.isqrt(quotient)
        if remainder or scale * scale != quotient or not 2 <= scale <= 5:
            continue
        if max(board.shape) * scale > 30:
            continue
        tiles = [_decode(g, p, scale, bg) for p in large]
        if any(t is None for t in tiles):
            continue
        yield dict(background=bg, carrier=carrier, board=board, holes=holes,
                   ports=ports, scale=scale, tiles=tiles, bounds=(r, y, c, x),
                   noise_sizes=tuple(map(len, noise)), locations=tuple(_bbox(p) for p in large))


class _SearchLimit(ValueError):
    pass


def _assemblies(info, use_seams=True, use_ports=True, limit=2):
    """Complete exact-cover search, allowing all D4 orientations of colored tiles.

    Early termination at two different colored boards is sufficient to prove
    ambiguity. A unique result is returned only after the search finishes.
    """
    board, holes, tiles = info['board'], info['holes'], info['tiles']
    h, w = board.shape
    positions = [tuple(map(int, p)) for p in np.argwhere(holes)]
    index = {p: i for i, p in enumerate(positions)}
    neighbors = []
    required = []
    for r, c in positions:
        neighbors.append(tuple(index[(r+dr, c+dc)] for dr, dc in _N4 if (r+dr, c+dc) in index))
        required.append(tuple(int(board[r+dr, c+dc]) for dr, dc in _N4
                              if info['ports'][r+dr, c+dc]))
    by_cell = [[] for _ in positions]
    for piece, tile in enumerate(tiles):
        for orientation in _orientations(tile):
            ph, pw = orientation.shape
            cells = [(int(r), int(c), int(orientation[r, c])) for r, c in np.argwhere(orientation >= 0)]
            for top in range(1, h-ph):
                for left in range(1, w-pw):
                    mapped = []
                    for r, c, color in cells:
                        p = (top+r, left+c)
                        if p not in index:
                            break
                        i = index[p]
                        if use_ports and any(color != v for v in required[i]):
                            break
                        mapped.append((i, color))
                    else:
                        bit = sum(1 << i for i, _ in mapped)
                        candidate = (piece, bit, tuple(mapped), orientation, (top, left))
                        for i, _ in mapped:
                            by_cell[i].append(candidate)
    full = (1 << len(positions)) - 1
    values = [-1] * len(positions)
    placements = [None] * len(tiles)
    outputs, nodes = {}, 0

    def visit(occupied, used):
        nonlocal nodes
        nodes += 1
        if nodes > 120000:
            raise _SearchLimit("exact-cover enumeration did not finish")
        if occupied == full:
            if used != (1 << len(tiles)) - 1:
                return
            answer = board.copy()
            for (r, c), color in zip(positions, values):
                answer[r, c] = color
            key = answer.tobytes()
            if key not in outputs:
                outputs[key] = (answer, tuple(placements))
            return
        free = full ^ occupied
        cell = (free & -free).bit_length() - 1
        for candidate in by_cell[cell]:
            piece, bit, mapped, orientation, origin = candidate
            if (used >> piece) & 1 or occupied & bit:
                continue
            if use_seams and any(values[j] >= 0 and values[j] != color
                                 for i, color in mapped for j in neighbors[i]):
                continue
            for i, color in mapped:
                values[i] = color
            placements[piece] = (orientation, origin)
            visit(occupied | bit, used | (1 << piece))
            for i, _ in mapped:
                values[i] = -1
            placements[piece] = None
            if len(outputs) >= limit:
                return

    visit(0, 0)
    return list(outputs.values()), nodes


def _recover(input_grid):
    g = _grid(input_grid)
    solutions = []
    for info in _interpretations(g):
        assemblies, nodes = _assemblies(info)
        if len(assemblies) > 1:
            raise ValueError("ambiguous colored mosaic")
        if assemblies:
            logical, placements = assemblies[0]
            result = np.repeat(np.repeat(logical, info['scale'], 0), info['scale'], 1)
            solutions.append((result, info, logical, placements, nodes))
    if len(solutions) != 1:
        raise ValueError("expected one board, scale, and colored mosaic")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the filled, enlarged blueprint from this input alone."""
    return _recover(input_grid)[0].copy()


def _partition(rng, holes, count):
    groups = [set(p) for p in _components(holes)]
    while len(groups) < count:
        choices = []
        for i, cells in enumerate(groups):
            for axis in (0, 1):
                low = min(p[axis] for p in cells)
                high = max(p[axis] for p in cells)
                for cut in range(low+1, high+1):
                    a = {p for p in cells if p[axis] < cut}
                    b = cells - a
                    if min(len(a), len(b)) < 3:
                        continue
                    ma, mb = np.zeros(holes.shape, bool), np.zeros(holes.shape, bool)
                    for p in a:
                        ma[p] = True
                    for p in b:
                        mb[p] = True
                    if len(_components(ma)) == len(_components(mb)) == 1:
                        choices.append((i, a, b, abs(len(a)-len(b))))
        if not choices:
            return None
        weights = np.array([len(groups[i]) / (1 + imbalance) for i, _, _, imbalance in choices])
        i, a, b, _ = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        groups[i:i+1] = [a, b]
    labels = np.full(holes.shape, -1, dtype=int)
    for i, cells in enumerate(groups):
        for p in cells:
            labels[p] = i
    # Transfer boundary cells to make complementary tabs without breaking objects.
    for _ in range(int(rng.integers(0, 5))):
        options = [(r, c, int(labels[r+dr, c+dc])) for r, c in np.argwhere(holes)
                   for dr, dc in _N4 if 0 <= r+dr < holes.shape[0] and 0 <= c+dc < holes.shape[1]
                   and labels[r+dr, c+dc] >= 0 and labels[r+dr, c+dc] != labels[r, c]]
        rng.shuffle(options)
        for r, c, recipient in options:
            donor = int(labels[r, c])
            if np.count_nonzero(labels == donor) <= 3:
                continue
            labels[r, c] = recipient
            if all(len(_components(labels == i)) == 1 for i in (donor, recipient)):
                break
            labels[r, c] = donor
    return labels


def _layout(rng, scale, chambers, count, irregular, symmetric=False):
    limit = 30 // scale
    if chambers == 1:
        ch = int(rng.integers(3, min(6, limit-2)+1))
        cw = int(rng.integers(3, min(6, limit-2)+1))
        if symmetric:
            cw = int(rng.choice([v for v in (3, 5) if v <= limit-2]))
        bh = int(rng.integers(ch+2, min(limit, ch+4)+1))
        bw = int(rng.integers(cw+2, min(limit, cw+4)+1))
        r = int(rng.integers(1, bh-ch))
        c = int(rng.integers(1, bw-cw))
        holes = np.zeros((bh, bw), bool)
        holes[r:r+ch, c:c+cw] = True
        if irregular:
            # A bite enters from the cavity boundary; it is board structure.
            side = int(rng.integers(2 if symmetric else 4))
            if side < 2:
                rr = r if side == 0 else r+ch-1
                cc = c+cw//2 if symmetric else int(rng.integers(c, c+cw))
            else:
                rr = int(rng.integers(r, r+ch))
                cc = c if side == 2 else c+cw-1
            holes[rr, cc] = False
            if ch >= 5 and cw >= 5 and rng.random() < .35:
                holes[r+ch//2, c+cw//2] = False
    else:
        if limit < 8:
            return None
        bh, bw = int(rng.integers(8, min(limit, 10)+1)), int(rng.integers(8, min(limit, 10)+1))
        holes = np.zeros((bh, bw), bool)
        holes[2:-2, 2:-2] = True
        if rng.integers(2):
            holes[bh//2, :] = False
        else:
            holes[:, bw//2] = False
        if irregular:
            choices = [(r, c) for r, c in np.argwhere(holes)
                       if r in (2, bh-3) or c in (2, bw-3)]
            holes[choices[int(rng.integers(len(choices)))]] = False
    area = int(holes.sum())
    if area < count*3 or area*scale*scale > 330:
        return None
    labels = _partition(rng, holes, count)
    if labels is None:
        return None
    return holes, labels


def _picture(rng, holes, labels, bg, carrier, palette):
    h, w = holes.shape
    picture = np.full((h, w), carrier, dtype=np.int8)
    # Large regions, bands, or bent walks are made before cutting.
    kind = int(rng.integers(3))
    if kind == 0:
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            picture[r:min(h,r+int(rng.integers(1,4))), c:min(w,c+int(rng.integers(1,4)))] = int(rng.choice(palette))
    elif kind == 1:
        axis = int(rng.integers(2))
        for i in range(h if axis == 0 else w):
            if axis == 0:
                picture[i, :] = int(rng.choice(palette))
            else:
                picture[:, i] = int(rng.choice(palette))
    else:
        for color in palette:
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(h+w):
                picture[r, c] = color
                dr, dc = _N4[int(rng.integers(4))]
                r, c = min(h-1,max(0,r+dr)), min(w-1,max(0,c+dc))
    parent = list(range(h*w))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r, c in np.argwhere(holes):
        for dr, dc in ((1, 0), (0, 1)):
            y, x = r+dr, c+dc
            if y < h and x < w and holes[y, x] and labels[r, c] != labels[y, x]:
                parent[root(int(y*w+x))] = root(int(r*w+c))
    groups = {}
    for r, c in np.argwhere(holes):
        groups.setdefault(root(int(r*w+c)), []).append((int(r), int(c)))
    for cells in groups.values():
        # Quiet matches coexist with repeated colored boundary runs.
        values = [int(picture[p]) for p in cells]
        color = int(rng.choice(values)) if rng.random() < .6 else int(rng.choice([carrier]+palette))
        for p in cells:
            picture[p] = color
    # Sparse port locations, all with exactly one blank neighbor.
    candidates = []
    for r in range(1, h-1):
        for c in range(1, w-1):
            if holes[r, c]:
                continue
            adjacent = [(r+dr, c+dc) for dr, dc in _N4 if holes[r+dr, c+dc]]
            if len(adjacent) == 1:
                candidates.append(((r, c), adjacent[0]))
    if len(candidates) < 2:
        return None
    rng.shuffle(candidates)
    nports = int(rng.integers(2, min(5, len(candidates))+1))
    board = np.full(holes.shape, carrier, dtype=np.int8)
    board[holes] = bg
    for port, inside in candidates[:nports]:
        group = groups[root(inside[0]*w+inside[1])]
        color = int(picture[inside])
        if color == carrier:
            color = int(rng.choice(palette))
            for p in group:
                picture[p] = color
        board[port] = color
    # If two ports touch the same union group, the last color remains consistent.
    for port, inside in candidates[:nports]:
        board[port] = picture[inside]
    logical = board.copy()
    logical[holes] = picture[holes]
    tiles = []
    for i in range(int(labels.max())+1):
        r, y, c, x = _bbox(np.argwhere(labels == i))
        tiles.append(np.where(labels[r:y, c:x] == i, picture[r:y, c:x], -1).astype(np.int8))
    return board, logical, tiles


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _scatter(rng, board, tiles, scale, bg, noise_count):
    objects = [board] + [np.repeat(np.repeat(t, scale, 0), scale, 1) for t in tiles]
    for attempt in range(16):
        shape = (int(rng.integers(22, 31)), int(rng.integers(22, 31))) if attempt < 10 else (30, 30)
        canvas = np.full(shape, bg, dtype=np.int8)
        blocked = np.zeros(shape, bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            a = objects[i]
            h, w = a.shape
            choices = _free_origins(blocked, h, w)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            view = canvas[r:r+h, c:c+w]
            view[a >= 0] = a[a >= 0]
            blocked[max(0,r-1):min(shape[0],r+h+1), max(0,c-1):min(shape[1],c+w+1)] = True
        else:
            for _ in range(noise_count):
                choices = _free_origins(blocked, 1, 1)
                if not len(choices):
                    break
                r, c = map(int, choices[int(rng.integers(len(choices)))])
                canvas[r, c] = int(rng.choice([v for v in range(10) if v != bg]))
                blocked[max(0,r-1):r+2, max(0,c-1):c+2] = True
            if np.count_nonzero(canvas == bg) <= canvas.size // 2:
                continue
            if rng.random() < .7:
                r, y, c, x = _bbox(np.argwhere(canvas != bg))
                r, c = max(0,r-int(rng.integers(4))), max(0,c-int(rng.integers(4)))
                y, x = min(shape[0],y+int(rng.integers(4))), min(shape[1],x+int(rng.integers(4)))
                crop = canvas[r:y, c:x]
                if np.count_nonzero(crop == bg) > crop.size // 2:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, scale, chambers, count, irregular, require_reflection=False, require_rotation=False,
               require_ports=False):
    for attempt in range(240):
        layout = _layout(rng, scale, chambers, count, irregular, symmetric=require_ports)
        if layout is None:
            continue
        holes, labels = layout
        colors = list(map(int, rng.permutation(10)))
        bg, carrier = colors[:2]
        palette = colors[2:2+int(rng.integers(2, 5))]
        picture = _picture(rng, holes, labels, bg, carrier, palette)
        if picture is None:
            continue
        board, logical, tiles = picture
        info = dict(board=board, holes=holes, ports=(board != bg) & (board != carrier), tiles=tiles)
        try:
            answers, _ = _assemblies(info)
            if len(answers) != 1 or not np.array_equal(answers[0][0], logical):
                continue
            # A must remain essential even when geometric fit AND ports are kept.
            ablated, _ = _assemblies(info, use_seams=False)
            if len(ablated) != 2:
                continue
            if require_ports:
                unanchored, _ = _assemblies(info, use_ports=False)
                if len(unanchored) != 2:
                    continue
        except _SearchLimit:
            continue
        loose = []
        reflected = False
        rotated = False
        for tile in tiles:
            choices = _orientations(tile)
            if require_reflection and not reflected:
                rotations = {(a.shape, a.tobytes()) for a in (np.rot90(tile,k) for k in range(4))}
                reflected_choices = [a for a in choices if (a.shape,a.tobytes()) not in rotations]
                if reflected_choices:
                    choices = reflected_choices
                    reflected = True
            elif require_rotation and not rotated:
                rotation_choices = [np.rot90(tile, k).copy() for k in range(1,4)
                                    if not np.array_equal(np.rot90(tile,k), tile)]
                if rotation_choices:
                    choices = rotation_choices
                    rotated = True
            loose.append(choices[int(rng.integers(len(choices)))])
        if require_reflection and not reflected:
            continue
        if require_rotation and not rotated:
            continue
        inp = _scatter(rng, board, loose, scale, bg, int(rng.integers(0, 6)))
        if inp is None:
            continue
        expected = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        try:
            predicted, recovered, _, placements, _ = _recover(inp)
        except ValueError:
            continue
        if not np.array_equal(predicted, expected):
            continue
        if require_reflection and not _features(recovered, placements)['reflection']:
            continue
        if require_rotation and not _features(recovered, placements)['rotation']:
            continue
        return {'input': inp, 'output': expected}
    raise RuntimeError("bounded mosaic construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    plans = []
    if regime == 0:
        schedules = ((2,3), (3,2), (2,4), (3,4), (2,5))
        a, b = schedules[int(rng.integers(len(schedules)))]
        plans = [(a, 1, int(rng.integers(3, 5)), i < 2 or rng.random() < .7) for i in range(nt)]
        tests = [(b, 1, 3 if b == 5 else int(rng.integers(3, 6 if b < 4 else 5)), rng.random() < .7) for _ in range(ne)]
    elif regime == 1:
        plans = [(2, 1, int(rng.integers(3, 5)), i < 2 or rng.random() < .7) for i in range(nt)]
        tests = [(2, 2, int(rng.integers(4, 7)), rng.random() < .7) for _ in range(ne)]
    else:
        plans = [(2, 1, 3 if i == 0 else 4, i < 2 or rng.random() < .7) for i in range(nt)]
        tests = [(2, 1, int(rng.integers(5, 7)), rng.random() < .7) for _ in range(ne)]
    # The first two examples show reflection and rotation of patterned pieces.
    puzzle, seen = {'train': [], 'test': []}, set()
    first_mask = None
    for split, split_plans in (('train', plans), ('test', tests)):
        for i, plan in enumerate(split_plans):
            for _ in range(10):
                pair = _make_pair(rng, *plan, require_reflection=(split == 'train' and i == 0),
                                  require_rotation=(split == 'train' and i == 1),
                                  require_ports=(split == 'train' and i == 0))
                key = (pair['input'].shape, pair['input'].tobytes())
                info = _recover(pair['input'])[1]
                mask = (info['holes'].shape, info['holes'].tobytes())
                if key not in seen and not (split == 'train' and i == 1 and mask == first_mask):
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        first_mask = mask
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, placements):
    reflection = rotation = stepped = seams = 0
    labels = np.full(info['board'].shape, -1, int)
    for i, (source, (oriented, (r, c))) in enumerate(zip(info['tiles'], placements)):
        ph, pw = oriented.shape
        labels[r:r+ph, c:c+pw][oriented >= 0] = i
        rotations = {(a.shape, a.tobytes()) for a in (np.rot90(source,k) for k in range(4))}
        key = (oriented.shape, oriented.tobytes())
        reflection += key not in rotations
        rotation += key in rotations and key != (source.shape, source.tobytes())
        stepped += np.count_nonzero(source >= 0) < source.size
    for r, c in np.argwhere(info['holes']):
        for dr, dc in ((1, 0), (0, 1)):
            y, x = r+dr, c+dc
            if info['holes'][y, x] and labels[r, c] != labels[y, x]:
                seams += 1
    hs = _components(info['holes'])
    nonrectangular = any(len(p) != (lambda b: (b[1]-b[0])*(b[3]-b[2]))(_bbox(p)) for p in hs)
    return dict(scale=info['scale'], board_shape=info['board'].shape,
                chambers=len(hs), tiles=len(info['tiles']), ports=int(info['ports'].sum()),
                reflection=int(reflection), rotation=int(rotation), stepped=int(stepped),
                seams=seams, nonrectangular=nonrectangular,
                palette=tuple(map(int, np.unique(info['board']))),
                mask=(info['holes'].shape, info['holes'].tobytes()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, port_witness = set(), [], False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            expected, info, logical, placements, _ = _recover(inp)
            assert np.array_equal(expected, out), "output differs from the unique input-derived mosaic"
            assert not np.any(logical == info['background'])
            assert sum(np.count_nonzero(t >= 0) for t in info['tiles']) == int(info['holes'].sum())
            assert np.array_equal(logical[~info['holes']], info['board'][~info['holes']])
            # Exact preservation of tile color counts after D4 transforms.
            for color in range(10):
                assert np.count_nonzero(logical[info['holes']] == color) == sum(np.count_nonzero(t == color) for t in info['tiles'])
            ablated, _ = _assemblies(info, use_seams=False)
            assert len(ablated) == 2, "tile seam clues are redundant"
            features = _features(info, placements)
            assert features['seams'] > 0
            if split == 'train':
                training.append(features)
                if not port_witness:
                    unanchored, _ = _assemblies(info, use_ports=False)
                    port_witness |= len(unanchored) == 2
    assert any(f['reflection'] for f in training), "reflection convention lacks a training witness"
    assert any(f['rotation'] for f in training), "rotation convention lacks a training witness"
    assert any(f['nonrectangular'] for f in training), "cavity geometry lacks a training witness"
    assert port_witness, "fixed ports lack a training witness"
    assert len({f['mask'] for f in training}) >= 2, "training must vary structure"


def self_test():
    """Bounded reproducibility, ablation, preservation and diversity checks."""
    fingerprints, inputs, outputs, masks, scales, counts, chambers = set(), set(), set(), set(), set(), set(), set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple((s, p['input'].shape, p['input'].tobytes(), p['output'].shape, p['output'].tobytes())
                            for s in ('train', 'test') for p in puzzle[s])
        fingerprints.add(fingerprint)
        for s in ('train', 'test'):
            for p in puzzle[s]:
                result, info, _, placements, _ = _recover(p['input'])
                assert np.array_equal(result, p['output'])
                f = _features(info, placements)
                features.append(f)
                inputs.add(p['input'].shape)
                outputs.add(p['output'].shape)
                masks.add(f['mask'])
                scales.add(f['scale'])
                counts.add(f['tiles'])
                chambers.add(f['chambers'])
    assert len(fingerprints) == 42
    assert len(inputs) >= 20 and len(outputs) >= 15 and len(masks) >= 80
    assert scales >= {2, 3, 4, 5} and counts >= {3, 4, 5, 6} and chambers >= {1, 2}
    assert any(f['stepped'] for f in features) and any(f['rotation'] for f in features)
    a, b = generate_puzzle(918273, 2, 1), generate_puzzle(918273, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    _, info, logical, _, _ = _recover(a['train'][0]['input'])
    reordered = dict(info, tiles=list(reversed(info['tiles'])))
    reordered_answers, _ = _assemblies(reordered)
    assert len(reordered_answers) == 1 and np.array_equal(reordered_answers[0][0], logical)
    # Corrupting a port with a color absent from the tiles must break the match.
    damaged_input = a['train'][0]['input'].copy()
    absent = next(c for c in range(10) if c not in (info['background'], info['carrier'])
                  and all(not np.any(t == c) for t in info['tiles']))
    pr, pc = map(int, np.argwhere(info['ports'])[0])
    r, y, c, x = info['bounds']
    damaged_input[r+pr, c+pc] = absent
    try:
        solve(damaged_input)
    except ValueError:
        pass
    else:
        raise AssertionError("unmatchable port accepted")
    damaged_input = a['train'][0]['input'].copy()
    outside = np.ones(damaged_input.shape, bool)
    outside[r:y, c:x] = False
    component = next(p for p in _components(outside & (damaged_input != info['background'])) if len(p) >= 4)
    damaged_input[component[0]] = info['background']
    try:
        solve(damaged_input)
    except ValueError:
        pass
    else:
        raise AssertionError("broken macrocell accepted")
    # Differently colored centers in otherwise interchangeable tiles make a
    # genuine tie; equal centers make their identity permutation harmless.
    diagnostic = np.zeros((24, 25), dtype=np.int8)
    board = np.full((7, 10), 2, dtype=np.int8)
    board[2:5, 2:8] = 0
    board[1, 2] = board[5, 7] = 3
    diagnostic[1:8, 1:11] = board
    for left, center in ((1,4), (12,5)):
        tile = np.full((3,3), 3, dtype=np.int8)
        tile[1,1] = center
        diagnostic[12:18, left:left+6] = np.repeat(np.repeat(tile,2,0),2,1)
    try:
        solve(diagnostic)
    except ValueError as exc:
        assert 'ambiguous' in str(exc)
    else:
        raise AssertionError("distinct colored tie accepted")
    diagnostic[diagnostic == 5] = 4
    assert solve(diagnostic).shape == (14,20)
    return dict(seeds=42, distinct_puzzles=len(fingerprints), input_shapes=len(inputs),
                output_shapes=len(outputs), cavity_masks=len(masks), scales=sorted(scales),
                tile_counts=sorted(counts), chamber_counts=sorted(chambers))


if __name__ == '__main__':
    print(self_test())
