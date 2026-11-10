"""Shadow repair: exact rectangular packing constrained by light observations.

Only standard Python and NumPy are used.  The solver reads the input, not the
constructor's intended placements.  A probe is a readout, never an obstacle.
"""

from collections import Counter
import json

import numpy as np


BACKGROUND = 0
PANEL = 5


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, values=None):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color = None if values is None else values[r, c]
        todo = [(r, c)]
        seen[r, c] = True
        for rr, cc in todo:
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not seen[nr, nc]
                        and (values is None or values[nr, nc] == color)):
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        groups.append(todo)
    return groups


def _box(cells):
    return (min(r for r, _ in cells), min(c for _, c in cells),
            max(r for r, _ in cells)+1, max(c for _, c in cells)+1)


def _path(source, cell):
    """Source-to-probe segment, excluding both source and probe."""
    r0, c0, r1, c1 = source
    r, c = cell
    if r0 <= r < r1 and c < c0:
        return [(r, cc) for cc in range(c0-1, c, -1)]
    if r0 <= r < r1 and c >= c1:
        return [(r, cc) for cc in range(c1, c)]
    if c0 <= c < c1 and r < r0:
        return [(rr, c) for rr in range(r0-1, r, -1)]
    if c0 <= c < c1 and r >= r1:
        return [(rr, c) for rr in range(r1, r)]
    raise ValueError("probe is outside the source's orthogonal lanes")


def _parse(a):
    cells = list(map(tuple, np.argwhere(a == PANEL)))
    if not cells:
        raise ValueError("missing glass panel")
    if len(_components(a == PANEL)) != 1:
        raise ValueError("glass carrier must remain connected")
    board = _box(cells)
    r0, c0, r1, c1 = board
    view = a[r0:r1, c0:c1]
    if not (np.all(view[0] == PANEL) and np.all(view[-1] == PANEL)
            and np.all(view[:, 0] == PANEL) and np.all(view[:, -1] == PANEL)):
        raise ValueError("panel needs an intact border")
    inside = np.zeros(a.shape, dtype=bool)
    inside[r0:r1, c0:c1] = True
    specials = inside & (a != BACKGROUND) & (a != PANEL)
    groups = _components(specials, a)
    sources = []
    for group in groups:
        b = _box(group)
        if min(b[2]-b[0], b[3]-b[1]) >= 2:
            if len(group) != (b[2]-b[0])*(b[3]-b[1]):
                raise ValueError("nonrectangular source")
            sources.append((b, int(a[group[0]])))
    if len(sources) != 1:
        raise ValueError("expected one solid source of height and width >= 2")
    source, light = sources[0]
    sr0, sc0, sr1, sc1 = source
    source_mask = np.zeros(a.shape, dtype=bool)
    source_mask[sr0:sr1, sc0:sc1] = True
    probes = [(int(r), int(c), int(a[r, c]))
              for r, c in np.argwhere(specials & ~source_mask)]
    dark_colors = {v for _, _, v in probes if v != light}
    if len(dark_colors) != 1 or not any(v == light for _, _, v in probes):
        raise ValueError("need visible light and dark readouts")
    opaque = dark_colors.pop()
    for r, c, _ in probes:
        if any(0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
               and specials[rr, cc]
               for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
            raise ValueError("readouts must be isolated from each other and source")
    holes = sorted((int(r), int(c)) for r, c in np.argwhere(inside & (a == BACKGROUND)))
    if not holes:
        raise ValueError("missing repair cells")
    index = {cell: i for i, cell in enumerate(holes)}
    observations = []
    for r, c, color in probes:
        mask = sum(1 << index[p] for p in _path(source, (r, c)) if p in index)
        observations.append((mask, color == light))
    pieces, exterior = [], []
    for group in _components(~inside & (a != BACKGROUND), a):
        b = _box(group)
        color = int(a[group[0]])
        h, w = b[2]-b[0], b[3]-b[1]
        if color in (PANEL, light) or len(group) != h*w or h*w < 2:
            raise ValueError("inventory must contain detached rectangular tiles of area >= 2")
        pieces.append((color, h, w))
        exterior.extend(group)
    if not pieces or not any(c == opaque for c, _, _ in pieces):
        raise ValueError("missing inventory or opaque tile")
    if not any(c != opaque for c, _, _ in pieces):
        raise ValueError("missing transparent tile")
    if sum(h*w for _, h, w in pieces) != len(holes):
        raise ValueError("tile inventory area does not match holes")
    return dict(board=board, source=source, light=light, opaque=opaque,
                holes=holes, pieces=pieces, exterior=exterior,
                observations=observations, probes=probes, inside=inside)


def _consistent(opaque_mask, observations):
    return all((not bool(opaque_mask & mask)) == lit for mask, lit in observations)


def _pack(holes, pieces, opaque, observations=(), limit=2, node_limit=80000):
    """Enumerate distinct cell colorings, ignoring permutations of equal tiles.

    Budget exhaustion is an error, never evidence of uniqueness.  The generated
    family stays small, but solving does not use its construction profiles.
    """
    index = {cell: i for i, cell in enumerate(holes)}
    n = len(holes)
    full = (1 << n)-1
    if sum(h*w for _, h, w in pieces) != n:
        return []
    types = sorted(Counter(pieces).items())
    descriptors = [d for d, _ in types]
    counts = [k for _, k in types]
    choices = [[] for _ in holes]
    forbidden = 0
    dark = []
    for mask, lit in observations:
        if lit:
            forbidden |= mask
        else:
            if not mask:
                return []
            dark.append(mask)
    for t, (color, h, w) in enumerate(descriptors):
        for r, c in holes:
            cells = [(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)]
            if not all(p in index for p in cells):
                continue
            indices = tuple(index[p] for p in cells)
            mask = sum(1 << i for i in indices)
            if color == opaque and mask & forbidden:
                continue
            option = (t, color, mask, indices)
            for i in indices:
                choices[i].append(option)
    painted = [-1]*n
    results, seen = [], set()
    nodes = 0

    def search(filled, opaque_mask):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise ValueError("exact-cover node budget exceeded")
        if len(results) >= limit:
            return
        # A dark readout can no longer be explained once its whole path is glass.
        if any(not (opaque_mask & m) and not (m & (full ^ filled)) for m in dark):
            return
        if filled == full:
            coloring = tuple(painted)
            if coloring not in seen:
                seen.add(coloring)
                results.append(coloring)
            return
        # Every legal cover of the first missing cell starts at that cell:
        # a rectangle starting earlier would overlap an already covered cell.
        i = ((full ^ filled) & -(full ^ filled)).bit_length()-1
        for t, color, mask, indices in choices[i]:
            if not counts[t] or filled & mask:
                continue
            counts[t] -= 1
            for k in indices:
                painted[k] = color
            search(filled | mask, opaque_mask | (mask if color == opaque else 0))
            for k in indices:
                painted[k] = -1
            counts[t] += 1
            if len(results) >= limit:
                return

    search(0, 0)
    return results


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _parse(a)
    answers = _pack(info['holes'], info['pieces'], info['opaque'],
                    info['observations'], node_limit=None)
    if len(answers) != 1:
        raise ValueError("repair is impossible or has multiple output colorings")
    out = a.copy()
    for p, color in zip(info['holes'], answers[0]):
        out[p] = color
    for p in info['exterior']:
        out[p] = BACKGROUND
    return out


def _cell_swap_witness(coloring, opaque, observations):
    """Remove whole-tile geometry but conserve every color's area."""
    omask = sum(1 << i for i, c in enumerate(coloring) if c == opaque)
    for i, ci in enumerate(coloring):
        if ci != opaque:
            continue
        for j, cj in enumerate(coloring):
            if cj != opaque and _consistent(omask ^ (1 << i) ^ (1 << j), observations):
                return i, j
    return None


def _side(source, r, c):
    r0, c0, r1, c1 = source
    if r0 <= r < r1:
        return 'left' if c < c0 else 'right'
    return 'top' if r < r0 else 'bottom'


def _probe_evidence(source, probes):
    selective = depth = False
    axes = set()
    for r, c, _, lit in probes:
        side = _side(source, r, c)
        axes.add('horizontal' if side in ('left', 'right') else 'vertical')
        for rr, cc, _, other_lit in probes:
            if side != _side(source, rr, cc) or lit == other_lit:
                continue
            same_lane = r == rr if side in ('left', 'right') else c == cc
            if same_lane:
                depth = True
            else:
                selective = True
    return selective, depth, axes


def _ray_readouts(a, b, info):
    """Check the repaired mosaic by forward, first-hit lane scans.

    Only placed opaque material blocks.  A dark readout is an observation,
    not a piece of opaque material, and therefore does not itself cast a shadow.
    """
    r0, c0, r1, c1 = info['source']
    br0, bc0, br1, bc1 = info['board']
    reached = set()
    lanes = []
    for r in range(r0, r1):
        lanes.append([(r, c) for c in range(c0-1, bc0, -1)])
        lanes.append([(r, c) for c in range(c1, bc1-1)])
    for c in range(c0, c1):
        lanes.append([(r, c) for r in range(r0-1, br0, -1)])
        lanes.append([(r, c) for r in range(r1, br1-1)])
    holes = set(info['holes'])
    for lane in lanes:
        for p in lane:
            if p in holes and b[p] == info['opaque']:
                break
            reached.add(p)
    return all(((r, c) in reached) == (color == info['light'])
               for r, c, color in info['probes'])


def _place_tiles(rng, a, board, pieces):
    r0, c0, r1, c1 = board
    reserved = np.zeros(a.shape, dtype=bool)
    reserved[max(0, r0-1):min(a.shape[0], r1+1),
             max(0, c0-1):min(a.shape[1], c1+1)] = True
    for color, h, w in sorted(pieces, key=lambda p: -p[1]*p[2]):
        candidates = []
        for r in range(a.shape[0]-h+1):
            for c in range(a.shape[1]-w+1):
                if not np.any(reserved[r:r+h, c:c+w]):
                    candidates.append((r, c))
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(len(candidates)))]
        a[r:r+h, c:c+w] = color
        reserved[max(0, r-1):min(a.shape[0], r+h+1),
                 max(0, c-1):min(a.shape[1], c+w+1)] = True
    return True


def _construct(rng, roles, test, index):
    light, opaque, glass = roles
    for attempt in range(400):
        sh, sw = map(int, rng.integers(2, 6, size=2))
        if test:
            if rng.random() < .5:
                sh = int(rng.integers(6, 9))
            else:
                sw = int(rng.integers(6, 9))
        ph = int(rng.integers(max(12, sh+6), min(21, sh+13)+1))
        pw = int(rng.integers(max(12, sw+6), min(21, sw+13)+1))
        sr = int(rng.integers(3, ph-sh-2))
        sc = int(rng.integers(3, pw-sw-2))
        source = (sr, sc, sr+sh, sc+sw)
        occupied = np.zeros((ph, pw), dtype=bool)
        occupied[sr-1:sr+sh+1, sc-1:sc+sw+1] = True
        # Two exchangeable shapes deliberately prevent shape-only decoding.
        h, w = map(int, rng.integers(1, 4, size=2))
        if h*w < 2:
            w = 2
        count = int(rng.integers(4, 9))
        opaque_count = 1 if index == 0 and not test else int(rng.integers(1, 4))
        opaque_count = min(opaque_count, count-2)
        dims = [(h, w), (h, w)]
        dims.extend(tuple(map(int, rng.integers(1, 5, size=2))) for _ in range(count-2))
        dims = [(hh, ww if hh*ww >= 2 else 2) for hh, ww in dims]
        colors = [opaque, glass] + [opaque]*(opaque_count-1) + [glass]*(count-opaque_count-1)
        rectangles = []
        good = True
        for k, ((hh, ww), color) in enumerate(zip(dims, colors)):
            candidates = []
            for r in range(2, ph-hh-1):
                for c in range(2, pw-ww-1):
                    if np.any(occupied[r:r+hh, c:c+ww]):
                        continue
                    # The equal slots must both be visible to at least one ray.
                    if k < 2 and not (max(r, sr) < min(r+hh, sr+sh)
                                      or max(c, sc) < min(c+ww, sc+sw)):
                        continue
                    touch = any((r+hh == rr or rr+rh == r) and max(c, cc) < min(c+ww, cc+rw)
                                or (c+ww == cc or cc+rw == c) and max(r, rr) < min(r+hh, rr+rh)
                                for rr, cc, rh, rw, _ in rectangles)
                    candidates.append((r, c, touch))
            if not candidates:
                good = False
                break
            touching = [p for p in candidates if p[2]]
            separate = [p for p in candidates if not p[2]]
            pool = (touching if touching and rng.random() < .65 else separate) or candidates
            r, c, _ = pool[int(rng.integers(len(pool)))]
            occupied[r:r+hh, c:c+ww] = True
            rectangles.append((r, c, hh, ww, color))
        if not good:
            continue
        # Establish destination colors independently of the exact-cover solver.
        destination = {(rr, cc): color for r, c, hh, ww, color in rectangles
                       for rr in range(r, r+hh) for cc in range(c, c+ww)}
        hole_cells = sorted(destination)
        if not test and index == 0:
            hole_mask = np.zeros((ph, pw), dtype=bool)
            for p in hole_cells:
                hole_mask[p] = True
            mixed_junction = False
            for group in _components(hole_mask):
                gr, gc, gh, gw = _box(group)
                mixed_junction |= (len(group) < (gh-gr)*(gw-gc)
                                   and len({destination[p] for p in group}) > 1)
            if not mixed_junction:
                continue
        intended = tuple(destination[p] for p in hole_cells)
        pieces = [(color, hh, ww) for _, _, hh, ww, color in rectangles]
        try:
            alternatives = _pack(hole_cells, pieces, opaque, limit=256)
        except ValueError:
            continue
        if not 2 <= len(alternatives) < 256 or intended not in alternatives:
            continue
        hi = {p: i for i, p in enumerate(hole_cells)}
        omask = sum(1 << i for i, color in enumerate(intended) if color == opaque)
        candidates = []
        for r in range(1, ph-1):
            for c in range(1, pw-1):
                if occupied[r, c]:
                    continue
                try:
                    path = _path(source, (r, c))
                except ValueError:
                    continue
                mask = sum(1 << hi[p] for p in path if p in hi)
                lit = not bool(mask & omask)
                candidates.append((r, c, mask, lit))
        rng.shuffle(candidates)
        remaining = list(alternatives)
        probes = []
        # The first two examples teach complementary optical facts.  Candidate
        # readings come from the intended mosaic; exact cover still verifies it.
        if not test and index < 2:
            pairs = []
            for i, p in enumerate(candidates):
                r, c, _, lit = p
                side = _side(source, r, c)
                for q in candidates[i+1:]:
                    rr, cc, _, other_lit = q
                    if (lit == other_lit or side != _side(source, rr, cc)
                            or abs(r-rr)+abs(c-cc) <= 1):
                        continue
                    same_lane = r == rr if side in ('left', 'right') else c == cc
                    if same_lane == (index == 1):
                        pairs.append((p, q))
            if not pairs:
                continue
            probes.extend(pairs[int(rng.integers(len(pairs)))])
            remaining = [v for v in remaining if _consistent(
                sum(1 << i for i, col in enumerate(v) if col == opaque),
                [(m, lit) for _, _, m, lit in probes])]
        # Choose observations that separate actual alternative packings.
        while len(remaining) > 1:
            scored = []
            for r, c, mask, lit in candidates:
                if any(abs(r-rr)+abs(c-cc) <= 1 for rr, cc, _, _ in probes):
                    continue
                keep = [v for v in remaining if _consistent(
                    sum(1 << i for i, col in enumerate(v) if col == opaque), [(mask, lit)])]
                if len(keep) < len(remaining):
                    scored.append((len(keep), r, c, mask, lit, keep))
            if not scored or len(probes) >= 10:
                break
            best_size = min(p[0] for p in scored)
            best = [p for p in scored if p[0] == best_size]
            _, r, c, mask, lit, remaining = best[int(rng.integers(len(best)))]
            probes.append((r, c, mask, lit))
        if len(remaining) != 1:
            continue
        # Show both material behaviors, including a reached path through glass.
        for wanted in (True, False):
            if any(p[3] == wanted and (not wanted or p[2]) for p in probes):
                continue
            pool = [p for p in candidates if p[3] == wanted
                    and (not wanted or p[2])
                    and not any(abs(p[0]-q[0])+abs(p[1]-q[1]) <= 1 for q in probes)]
            if pool:
                probes.append(pool[int(rng.integers(len(pool)))])
        if {p[3] for p in probes} != {True, False}:
            continue
        if not any(lit and mask for _, _, mask, lit in probes):
            continue
        if not test and index == 0:
            # Expose all four sides, so a test need not introduce an unobserved
            # emission direction.  Some clear readouts can have no hole en route.
            for side in ('left', 'right', 'top', 'bottom'):
                if any(_side(source, p[0], p[1]) == side for p in probes):
                    continue
                pool = [p for p in candidates if _side(source, p[0], p[1]) == side
                        and not any(abs(p[0]-q[0])+abs(p[1]-q[1]) <= 1 for q in probes)]
                if not pool:
                    break
                probes.append(pool[int(rng.integers(len(pool)))])
            if len({_side(source, p[0], p[1]) for p in probes}) != 4:
                continue
        observations = [(m, lit) for _, _, m, lit in probes]
        if _cell_swap_witness(intended, opaque, observations) is None:
            continue
        local = np.full((ph, pw), PANEL, dtype=np.int8)
        local[sr:sr+sh, sc:sc+sw] = light
        for p in hole_cells:
            local[p] = BACKGROUND
        for r, c, _, lit in probes:
            local[r, c] = light if lit else opaque
        # A connected carrier and intact border make the object parsing explicit.
        if len(_components(local == PANEL)) != 1:
            continue
        height = int(rng.integers(max(ph+3, 18), 31))
        width = int(rng.integers(max(pw+3, 18), 31))
        top = int(rng.integers(1, height-ph))
        left = int(rng.integers(1, width-pw))
        board = (top, left, top+ph, left+pw)
        a = np.zeros((height, width), dtype=np.int8)
        a[top:top+ph, left:left+pw] = local
        if not _place_tiles(rng, a, board, pieces):
            continue
        out = np.zeros_like(a)
        out[top:top+ph, left:left+pw] = local
        for (r, c), color in destination.items():
            out[top+r, left+c] = color
        rotation = int(rng.integers(4))
        if rng.random() < .5:
            a, out = np.fliplr(a), np.fliplr(out)
        a, out = np.rot90(a, rotation).copy(), np.rot90(out, rotation).copy()
        # Parse the returned pixels and reject any accidental role ambiguity.
        try:
            info = _parse(a)
            answers = _pack(info['holes'], info['pieces'], opaque, info['observations'])
        except ValueError:
            continue
        expected = tuple(int(out[p]) for p in info['holes'])
        if answers == [expected]:
            return dict(input=a, output=out)
    raise RuntimeError("bounded construction could not find a uniquely observable repair")


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not low <= int(v) <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice([1, 2, 3, 4, 6, 7, 8, 9], 3, replace=False)))
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(8):
                pair = _construct(rng, roles, split == 'test', index)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("duplicate construction")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, roles = set(), set()
    for split in ('train', 'test'):
        for pair_index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            info = _parse(a)
            opaque = info['opaque']
            transparent = {c for c, _, _ in info['pieces'] if c != opaque}
            roles.add((info['light'], opaque, tuple(sorted(transparent))))
            assert np.all(b[~info['inside']] == BACKGROUND)
            fixed = info['inside'] & (a != BACKGROUND)
            assert np.array_equal(a[fixed], b[fixed])
            coloring = tuple(int(b[p]) for p in info['holes'])
            inventory = Counter()
            for c, h, w in info['pieces']:
                inventory[c] += h*w
            assert Counter(coloring) == inventory
            assert _pack(info['holes'], info['pieces'], opaque,
                         info['observations']) == [coloring]
            assert _consistent(sum(1 << i for i, c in enumerate(coloring) if c == opaque),
                               info['observations'])
            assert _ray_readouts(a, b, info)
            assert any(lit and mask for mask, lit in info['observations'])
            # Both parent ablations leave a different possible answer.
            assert len(_pack(info['holes'], info['pieces'], opaque, limit=2)) == 2
            assert _cell_swap_witness(coloring, opaque, info['observations']) is not None
            assert np.array_equal(solve(a), b)
            sh = info['source'][2]-info['source'][0]
            sw = info['source'][3]-info['source'][1]
            assert (max(sh, sw) >= 6) == (split == 'test')
            probes = [(r, c, 0, color == info['light']) for r, c, color in info['probes']]
            selective, depth, axes = _probe_evidence(info['source'], probes)
            if split == 'train' and pair_index == 0:
                assert selective and len(axes) == 2
                assert len({_side(info['source'], r, c) for r, c, _ in info['probes']}) == 4
                mixed_junction = False
                for group in _components(info['inside'] & (a == BACKGROUND)):
                    r0, c0, r1, c1 = _box(group)
                    mixed_junction |= (len(group) < (r1-r0)*(c1-c0)
                                       and len({int(b[cell]) for cell in group}) > 1)
                assert mixed_junction
            if split == 'train' and pair_index == 1:
                assert depth
    assert len(roles) == 1


def self_test():
    stats = Counter()
    shapes, sources, inventory_sizes, normalized, role_sets = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                info = _parse(a)
                shapes.add(a.shape)
                source = info['source']
                sources.add((source[2]-source[0], source[3]-source[1]))
                inventory_sizes.add(len(info['pieces']))
                role_sets.add((info['light'], info['opaque']))
                mapping = {0: 0, 5: 5, info['light']: 1, info['opaque']: 2}
                normalized.add((a.shape, bytes(mapping.get(int(c), 3) for c in a.flat)))
                stats['pairs'] += 1
                stats['non_square_tile'] += any(h != w for _, h, w in info['pieces'])
                stats['multiple_opaque_tiles'] += sum(c == info['opaque'] for c, _, _ in info['pieces']) > 1
                for component in _components(info['inside'] & (a == BACKGROUND)):
                    rb, cb, re, ce = _box(component)
                    stats['nonrectangular_opening'] += len(component) < (re-rb)*(ce-cb)
                    stats['mixed_opening'] += len({int(b[cell]) for cell in component}) > 1
                stats['horizontal_probe'] += any(source[0] <= r < source[2] for r, _, _ in info['probes'])
                stats['vertical_probe'] += any(source[1] <= c < source[3] for _, c, _ in info['probes'])
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p
                       for x, y in zip(p[s], repeat[s]) for k in ('input', 'output'))
            for pair in p['train'] + p['test']:
                a, b = pair['input'], pair['output']
                before = a.copy()
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
                assert np.array_equal(a, before)
    assert len(shapes) >= 30 and len(sources) >= 20 and len(inventory_sizes) >= 3
    assert len(normalized) == stats['pairs'] and len(role_sets) >= 20
    for key in ('non_square_tile', 'multiple_opaque_tiles', 'nonrectangular_opening',
                'mixed_opening', 'horizontal_probe', 'vertical_probe'):
        assert stats[key] >= 15, (key, stats[key])
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            _counts(train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, grid_shapes=len(shapes), source_shapes=len(sources),
                normalized_inputs=len(normalized), inventory_counts=sorted(inventory_sizes),
                measurements=dict(stats))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
