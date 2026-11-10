"""Weld and outline the marker's connected island of nonmodal motif tiles.

Only standard-library modules and NumPy are used. No parent code is imported.
The marker supplies the contour ink, so solve needs only the query input.
"""
from collections import Counter
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
           if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer rectangles, size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _flood(allowed, start):
    h, w = allowed.shape
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    stack = [start]
    reached[start] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and allowed[nr, nc]
                    and not reached[nr, nc]):
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        start = tuple(map(int, np.argwhere(remaining)[0]))
        group = _flood(remaining, start)
        result.append(group)
        remaining[group] = False
    return result


def _neighbors(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask, dtype=bool)
    for dr, dc in N8 if diagonal else N4:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _assemble(tiles, lattice):
    nr, nc, ph, pw = tiles.shape
    out = np.full((nr * (ph + 1) + 1, nc * (pw + 1) + 1),
                  lattice, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = 1 + r * (ph + 1), 1 + c * (pw + 1)
            out[y:y + ph, x:x + pw] = tiles[r, c]
    return out


def _decode(grid):
    """Recover all roles and tile identities, after undoing the carrier marker."""
    g = _grid(grid)
    lattice = int(g[0, 0])
    rs = np.flatnonzero(np.all(g == lattice, axis=1))
    cs = np.flatnonzero(np.all(g == lattice, axis=0))
    if (len(rs) < 3 or len(cs) < 3 or rs[0] != 0 or cs[0] != 0
            or rs[-1] != g.shape[0] - 1 or cs[-1] != g.shape[1] - 1
            or not np.all(np.diff(rs) == rs[1])
            or not np.all(np.diff(cs) == cs[1])):
        raise ValueError("a regular framed tile lattice is required")
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    if ph < 3 or pw < 3:
        raise ValueError("tiles need an interior and a one-cell carrier margin")
    values, counts = np.unique(g, return_counts=True)
    singles = values[counts == 1]
    if len(values) != 4 or len(singles) != 1:
        raise ValueError("four visible roles and exactly one singleton marker required")
    ink = int(singles[0])
    marker = tuple(map(int, np.argwhere(g == ink)[0]))
    nr, nc = len(rs) - 1, len(cs) - 1
    tiles = np.empty((nr, nc, ph, pw), dtype=np.int8)
    margins = []
    for r in range(nr):
        for c in range(nc):
            p = g[rs[r] + 1:rs[r + 1], cs[c] + 1:cs[c + 1]].copy()
            tiles[r, c] = p
            margins.extend(int(v) for v in p[[0, -1], :].flat if v != ink)
            margins.extend(int(v) for v in p[:, [0, -1]].flat if v != ink)
    if len(set(margins)) != 1 or lattice in margins or ink in margins:
        raise ValueError("all tile perimeters must share the carrier role")
    carrier = margins[0]
    motif = next(int(v) for v in values if v not in (lattice, ink, carrier))
    tiles[tiles == ink] = carrier
    if not np.all((tiles == carrier) | (tiles == motif)):
        raise ValueError("tile contents must use carrier and motif only")
    keys = [_key(p) for row in tiles for p in row]
    frequencies = Counter(keys)
    maximum = max(frequencies.values())
    winners = [k for k, v in frequencies.items() if v == maximum]
    if len(winners) != 1 or maximum < 2:
        raise ValueError("the master must be the unique repeated modal bitmap")
    master = next(p.copy() for row in tiles for p in row if _key(p) == winners[0])
    if set(map(int, np.unique(master))) != {carrier, motif}:
        raise ValueError("the master must visibly contain both interior roles")
    defects = np.any(tiles != master, axis=(2, 3))
    sr, sc = (marker[0] - 1) // (ph + 1), (marker[1] - 1) // (pw + 1)
    if not (0 <= sr < nr and 0 <= sc < nc and defects[sr, sc]
            and (marker[0] - 1) % (ph + 1) < ph
            and (marker[1] - 1) % (pw + 1) < pw):
        raise ValueError("the marker must replace a carrier cell in an altered tile")
    selected = _flood(defects, (sr, sc))
    return dict(grid=g, tiles=tiles, master=master, defects=defects,
                selected=selected, marker=marker, seed_tile=(sr, sc),
                lattice=lattice, carrier=carrier, motif=motif, ink=ink,
                ph=ph, pw=pw, nr=nr, nc=nc)


def _stitched(selected, ph, pw):
    """A separator is owned only when every incident tile is selected."""
    nr, nc = selected.shape
    region = np.zeros((nr * (ph + 1) + 1, nc * (pw + 1) + 1), bool)
    for r, c in np.argwhere(selected):
        y, x = 1 + int(r) * (ph + 1), 1 + int(c) * (pw + 1)
        region[y:y + ph, x:x + pw] = True
        if c + 1 < nc and selected[r, c + 1]:
            region[y:y + ph, x + pw] = True
        if r + 1 < nr and selected[r + 1, c]:
            region[y + ph, x:x + pw] = True
        if (r + 1 < nr and c + 1 < nc
                and np.all(selected[r:r + 2, c:c + 2])):
            region[y + ph, x + pw] = True
    return region


def _boundary(region):
    outside = _flood(np.pad(~region, 1, constant_values=True), (0, 0))
    rim8 = _neighbors(outside, True)[1:-1, 1:-1] & region
    rim4 = _neighbors(outside)[1:-1, 1:-1] & region
    return outside[1:-1, 1:-1], rim8, rim4


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only template comparison, tile flood, seam welding, external rim."""
    info = _decode(input_grid)
    tiles = info['tiles'].copy()
    tiles[info['defects'] & ~info['selected']] = info['master']
    out = _assemble(tiles, info['lattice'])
    region = _stitched(info['selected'], info['ph'], info['pw'])
    # Tile rectangles already contain their bitmaps. Only owned lattice pixels
    # are changed to carrier; no motif or anomalous core cell is overwritten.
    out[region & (out == info['lattice'])] = info['carrier']
    _, rim, _ = _boundary(region)
    if np.any(out[rim] != info['carrier']):
        raise ValueError("the external rim must lie within the carrier margins")
    out[rim] = info['ink']
    out[info['marker']] = info['ink']
    return out


# Geometry is a sampling preference. The solver accepts all regular legal
# lattices satisfying the documented rule, including geometry absent here.
GEOMETRIES = tuple((nr, nc, ph, pw)
                   for nr in range(2, 8) for nc in range(2, 8)
                   for ph in range(3, 9) for pw in range(3, 9)
                   if nr * (ph + 1) + 1 <= 30
                   and nc * (pw + 1) + 1 <= 30 and nr * nc >= 8)
WITNESS_GEOMETRIES = tuple(g for g in GEOMETRIES if 5 <= min(g[:2]) <= max(g[:2]) <= 6
                           and (g[2] - 2) * (g[3] - 2) >= 2)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _grow(rng, allowed, target, start=None):
    cells = np.argwhere(allowed)
    if not len(cells):
        return np.zeros_like(allowed)
    start = tuple(map(int, _pick(rng, cells))) if start is None else start
    grown = np.zeros_like(allowed)
    grown[start] = True
    for _ in range(target - 1):
        frontier = np.argwhere(allowed & ~grown & _neighbors(grown))
        if not len(frontier):
            break
        grown[tuple(map(int, _pick(rng, frontier)))] = True
    return grown


def _topology(rng, nr, nc, kind):
    selected = np.zeros((nr, nc), bool)
    other = np.zeros_like(selected)
    if kind == 'hole_witness':
        r, c = int(rng.integers(nr - 2)), int(rng.integers(nc - 2))
        selected[r:r + 3, c:c + 3] = True
        selected[r + 1, c + 1] = False
        forbidden = np.zeros_like(selected)
        forbidden[r + 1, c + 1] = True
        # An exterior tail gives a concavity, teaching diagonal-only rim cells.
        for _ in range(int(rng.integers(1, 3))):
            frontier = np.argwhere(~selected & ~forbidden & _neighbors(selected))
            selected[tuple(map(int, _pick(rng, frontier)))] = True
        available = ~selected & ~_neighbors(selected) & ~forbidden
        other = _grow(rng, available, int(rng.integers(1, 4)))
    elif kind == 'small_witness':
        # A small branch and a strictly larger disconnected patch. Rotations
        # and reflections subsequently move them to every side of the canvas.
        r, c = int(rng.integers(nr - 3)), int(rng.integers(nc - 4))
        selected[r, c] = selected[r + 1, c] = selected[r + 1, c + 1] = True
        # Exactly a diagonal contact between the two patches teaches that
        # corner contact does not join tile components (unlike rim contact).
        other[r + 2:r + 4, c + 2:c + 5] = True
    else:
        style = int(rng.integers(4))
        if style == 0 and min(nr, nc) >= 4:
            rh, cw = int(rng.integers(3, nr)), int(rng.integers(3, nc))
            r, c = int(rng.integers(nr - rh + 1)), int(rng.integers(nc - cw + 1))
            selected[r:r + rh, c:c + cw] = True
            selected[r + 1:r + rh - 1, c + 1:c + cw - 1] = False
        elif style == 1:
            rh, cw = int(rng.integers(1, nr)), int(rng.integers(1, nc))
            r, c = int(rng.integers(nr - rh + 1)), int(rng.integers(nc - cw + 1))
            selected[r:r + rh, c:c + cw] = True
        else:
            selected = _grow(rng, np.ones_like(selected),
                             int(rng.integers(2, max(3, nr * nc // 2))))
        for _ in range(int(rng.integers(4))):
            available = ~(selected | other | _neighbors(selected | other))
            other |= _grow(rng, available, int(rng.integers(1, 5)))
    # Keep enough repeated templates to recognize the source of the anomalies.
    if not selected.any() or np.count_nonzero(~(selected | other)) < 3:
        return None
    if kind.endswith('witness') and not other.any():
        return None
    k = int(rng.integers(4))
    selected, other = np.rot90(selected, k), np.rot90(other, k)
    if bool(rng.integers(2)):
        selected, other = np.fliplr(selected), np.fliplr(other)
    return selected.copy(), other.copy()


def _master(rng, ph, pw, carrier, motif):
    h, w = ph - 2, pw - 2
    rr, cc = np.indices((h, w))
    style = int(rng.integers(5))
    if style == 0:
        mask = rng.random((h, w)) < float(rng.uniform(.25, .75))
    elif style == 1:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif style == 2:
        mask = _grow(rng, np.ones((h, w), bool), max(1, h * w // 2))
    elif style == 3:
        mask = (rr == 0) | (cc == 0) | (rr == h - 1) | (cc == w - 1)
    else:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    if h * w == 1:
        mask[:] = True
    else:
        if not mask.any():
            mask.flat[int(rng.integers(mask.size))] = True
        if mask.all():
            mask.flat[int(rng.integers(mask.size))] = False
    out = np.full((ph, pw), carrier, dtype=np.int8)
    out[1:-1, 1:-1][mask] = motif
    return out


def _variant(rng, master, carrier, motif, mode):
    core = master[1:-1, 1:-1].copy()
    ink, blank = np.flatnonzero(core == motif), np.flatnonzero(core == carrier)
    if mode == 'balance' and len(ink) and len(blank):
        n = int(rng.integers(1, min(len(ink), len(blank), max(1, core.size // 3)) + 1))
        core.flat[rng.choice(ink, n, replace=False)] = carrier
        core.flat[rng.choice(blank, n, replace=False)] = motif
    elif mode == 'add' and len(blank):
        n = int(rng.integers(1, min(len(blank), max(1, core.size // 2)) + 1))
        core.flat[rng.choice(blank, n, replace=False)] = motif
    elif mode == 'remove' and len(ink):
        n = int(rng.integers(1, min(len(ink), max(1, core.size // 2)) + 1))
        core.flat[rng.choice(ink, n, replace=False)] = carrier
    else:
        indices = rng.choice(core.size, int(rng.integers(1, max(2, core.size // 2 + 1))), replace=False)
        core.flat[indices] = np.where(core.flat[indices] == carrier, motif, carrier)
    out = master.copy()
    out[1:-1, 1:-1] = core
    return out


def _make_pair(rng, geometry, palette, kind, force_fallback=False):
    nr, nc, ph, pw = geometry
    lattice, carrier, motif, ink = palette
    for attempt in range(80 if force_fallback else 0, 81):
        if attempt == 80:
            # Constructive termination: enough master copies outnumber every
            # possible variant, with all required witness mechanisms present.
            a, b = np.zeros((nr, nc), bool), np.zeros((nr, nc), bool)
            if kind == 'hole_witness':
                a[1:4, 1:4] = True
                a[2, 2] = False
                a[0, 1] = True
                b[-1, -1] = True
            elif kind == 'small_witness':
                a[0, 0] = a[1, 0] = a[1, 1] = True
                b[2:4, 2:5] = True
            else:
                a[0, 0] = a[1, 0] = a[1, 1] = True
            topology = a, b
        else:
            topology = _topology(rng, nr, nc, kind)
        if topology is None:
            continue
        selected, other = topology
        # Topology transforms also transform tile axes, preserving the bounds.
        tr, tc = selected.shape
        if tr != nr:
            th, tw = pw, ph
        else:
            th, tw = ph, pw
        if tr * (th + 1) + 1 > 30 or tc * (tw + 1) + 1 > 30:
            continue
        master = _master(rng, th, tw, carrier, motif)
        tiles = np.tile(master, (tr, tc, 1, 1))
        positions = list(map(tuple, np.argwhere(selected)))
        rng.shuffle(positions)
        for i, p in enumerate(positions):
            mode = ('add', 'remove', 'balance')[i % 3] if kind == 'hole_witness' else _pick(rng, ('add', 'remove', 'balance', 'flip'))
            tiles[p] = _variant(rng, master, carrier, motif, mode)
        for p in map(tuple, np.argwhere(other)):
            tiles[p] = _variant(rng, master, carrier, motif, _pick(rng, ('add', 'remove', 'balance', 'flip')))
        counts = Counter(_key(p) for row in tiles for p in row)
        if counts[_key(master)] != max(counts.values()) or list(counts.values()).count(max(counts.values())) != 1:
            continue
        inp = _assemble(tiles, lattice)
        region = _stitched(selected, th, tw)
        _, rim, _ = _boundary(region)
        candidates = []
        for r, c in positions:
            y, x = 1 + int(r) * (th + 1), 1 + int(c) * (tw + 1)
            for a, b in np.argwhere(tiles[r, c] == carrier):
                pixel = (y + int(a), x + int(b))
                if kind == 'small_witness' and not rim[pixel]:
                    continue
                if kind == 'hole_witness' and rim[pixel]:
                    continue
                candidates.append(pixel)
        if not candidates:
            continue
        inp[_pick(rng, candidates)] = ink
        pair = {'input': inp, 'output': solve(inp)}
        f = _features(pair)
        if kind == 'hole_witness' and not (f['holes'] and f['balanced'] and f['additions'] and f['removals'] and f['diagonal']):
            continue
        if kind == 'small_witness' and not f['larger_distractor']:
            continue
        return pair
    raise RuntimeError("bounded scene sampling exhausted")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shared_palette = bool(rng.integers(2))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    # Two witnesses always teach all mechanisms, even with only two examples.
    first = _pick(rng, WITNESS_GEOMETRIES)
    choices = [g for g in WITNESS_GEOMETRIES if g[2:] != first[2:]]
    second = _pick(rng, choices)
    geometries = [first, second]
    for _ in range(nt - 2):
        geometries.append(_pick(rng, [g for g in GEOMETRIES if max(g[:2]) <= 6 and max(g[2:]) <= 5]))
    # A visible shift: either more tile positions or larger motif interiors.
    # These are preferences only; the same rule accepts either geometry in train.
    scale_shift = bool(rng.integers(2))
    largest_training_tile_axis = max(max(g[2:]) for g in geometries)
    test_choices = ([g for g in GEOMETRIES if max(g[2:]) > largest_training_tile_axis]
                    if scale_shift else [g for g in GEOMETRIES if 7 in g[:2]])
    geometries.extend(_pick(rng, test_choices) for _ in range(ne))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, geometry in enumerate(geometries):
        split = 'train' if i < nt else 'test'
        kind = 'hole_witness' if i == 0 else 'small_witness' if i == 1 else 'mixed'
        for _ in range(20):
            roles = palette if shared_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            pair = _make_pair(rng, geometry, roles, kind)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("input uniqueness sampling exhausted")
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _decode(pair['input'])
    region = _stitched(info['selected'], info['ph'], info['pw'])
    exterior, rim, orth = _boundary(region)
    holes = 0
    for r, c in np.argwhere(~info['selected']):
        y, x = 1 + int(r) * (info['ph'] + 1), 1 + int(c) * (info['pw'] + 1)
        holes += int(not exterior[y, x])
    edits = []
    for r, c in np.argwhere(info['selected']):
        panel = info['tiles'][r, c]
        add = int(np.count_nonzero((info['master'] == info['carrier']) & (panel == info['motif'])))
        remove = int(np.count_nonzero((info['master'] == info['motif']) & (panel == info['carrier'])))
        edits.append((add, remove))
    selected_area = int(info['selected'].sum())
    groups = _components(info['defects'])
    return dict(geometry=(info['nr'], info['nc'], info['ph'], info['pw']),
                palette=(info['lattice'], info['carrier'], info['motif'], info['ink']),
                master=_key(info['master'] == info['motif']),
                defect_map=_key(info['defects']), selected_map=_key(info['selected']),
                selected=selected_area, erased=int((info['defects'] & ~info['selected']).sum()),
                groups=len(groups), holes=holes, diagonal=int((rim & ~orth).sum()),
                balanced=any(a == b and a > 0 for a, b in edits),
                additions=any(a > 0 and b == 0 for a, b in edits),
                removals=any(b > 0 and a == 0 for a, b in edits),
                edit_sizes=tuple(a + b for a, b in edits),
                larger_distractor=any(int(m.sum()) > selected_area for m in groups),
                diagonal_patch_contact=bool(np.any(info['defects'] & ~info['selected']
                    & _neighbors(info['selected'], True))),
                welded=int(np.count_nonzero(region & (info['grid'] == info['lattice']))),
                marker_on_rim=bool(rim[info['marker']]),
                complete_junctions=int(np.count_nonzero(info['selected'][:-1, :-1]
                    & info['selected'][1:, :-1] & info['selected'][:-1, 1:]
                    & info['selected'][1:, 1:])))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count('num_train', nt, 2, 10)
    _count('num_test', ne, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("pairs must have exactly input and output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate inputs'
            seen.add(_key(inp))
            info = _decode(inp)
            assert out.shape == inp.shape and np.array_equal(out, solve(inp)), 'incorrect output'
            # Verify exact motif retention/restoration separately from the rim.
            for r in range(info['nr']):
                for c in range(info['nc']):
                    y, x = 1 + r * (info['ph'] + 1), 1 + c * (info['pw'] + 1)
                    core = out[y + 1:y + info['ph'] - 1, x + 1:x + info['pw'] - 1].copy()
                    my, mx = info['marker']
                    if y < my < y + info['ph'] - 1 and x < mx < x + info['pw'] - 1:
                        core[my - y - 1, mx - x - 1] = info['carrier']
                    source = info['tiles'][r, c] if info['selected'][r, c] else info['master']
                    assert np.array_equal(core, source[1:-1, 1:-1]), 'core was repaired or recolored'
            assert np.all(out[[0, -1], :] == info['lattice'])
            assert np.all(out[:, [0, -1]] == info['lattice'])
            assert out[info['marker']] == info['ink']
            f = _features(pair)
            assert f['selected'] > 0
            if split == 'train':
                train.append(f)
    for feature in ('holes', 'diagonal', 'balanced', 'additions', 'removals',
                    'erased', 'larger_distractor', 'diagonal_patch_contact',
                    'welded', 'marker_on_rim'):
        assert any(f[feature] for f in train), 'missing demonstration of ' + feature
    assert any(not f['marker_on_rim'] for f in train), 'demonstrate an interior seed'
    assert len({f['selected_map'] for f in train}) >= 2, 'demonstrate different selected shapes'


def _fixture(selected, other=(), side=4):
    master = np.full((side, side), 2, np.int8)
    master[1, 1] = 3
    if side == 4:
        master[2, 2] = 3
    tiles = np.tile(master, (*selected.shape, 1, 1))
    variant = master.copy()
    variant[1, 1] = 2
    if side == 4:
        variant[1, 2] = 3  # A count-preserving shape change.
    for p in map(tuple, np.argwhere(selected)):
        tiles[p] = variant
    for p in other:
        tiles[p] = variant
    inp = _assemble(tiles, 1)
    r, c = map(int, np.argwhere(selected)[0])
    inp[1 + r * (side + 1) + 2, 1 + c * (side + 1) + 1] = 4
    return inp


def _semantic_tests():
    # Concrete expected cells, including a diagonal-only concavity.
    selected = np.zeros((3, 4), bool)
    selected[0, 0] = selected[1, 0] = selected[1, 1] = True
    inp = _fixture(selected, ((0, 3),))
    out = solve(inp)
    assert out[6, 4] == 4  # Orthogonal neighbors are inside; diagonal is outside.
    assert out[5, 5] == 1  # Three selected incident tiles do not own the junction.
    assert out[8, 5] == 2  # Internal seam is welded, rather than outlined.
    assert out[2, 17] == 3 and inp[2, 17] == 2  # Distractor restored to master.
    assert out[7, 8] == 3 and out[7, 7] == 2  # Balanced edit copied exactly.
    ring = np.zeros((5, 5), bool)
    ring[1:4, 1:4] = True
    ring[2, 2] = False
    hole_in = _fixture(ring, ((4, 4),))
    hole_out = solve(hole_in)
    assert hole_out[9, 12] == 2  # Tile margin facing the enclosed hole is unoutlined.
    assert hole_out[6, 12] == 4  # Same tile's outside-facing edge is outlined.
    assert hole_out[10, 12] == 1 and hole_out[12, 12] == 3  # Hole seam/master intact.
    assert hole_out[8, 10] == 2  # Seam between retained anomalies removed.
    block = np.zeros((4, 4), bool)
    block[:2, :2] = True
    assert solve(_fixture(block))[5, 5] == 2  # Four-way internal junction owned.
    perforated = np.zeros((7, 7), bool)
    perforated[1:6, 1:6] = True
    perforated[2, 2] = perforated[4, 4] = False
    multi_in = _fixture(perforated, side=3)
    multi_out = solve(multi_in)
    assert multi_out[10, 10] == multi_out[18, 18] == 3
    assert multi_out[10, 7] == 2 and multi_out[10, 5] == 4
    nested = np.zeros((7, 7), bool)
    nested[1:6, 1:6] = True
    nested[2:5, 2:5] = False
    nested_in = _fixture(nested, ((3, 3),), side=3)
    nested_out = solve(nested_in)
    assert nested_in[14, 14] == 2 and nested_out[14, 14] == 3
    assert nested_out[7, 14] == 2 and nested_out[5, 14] == 4
    for g in (inp, hole_in, _fixture(block)):
        expected = solve(g)
        before = g.copy()
        for k in range(4):
            for flip in (False, True):
                a, b = np.rot90(g, k), np.rot90(expected, k)
                if flip:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b)
        lookup = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], np.int8)
        assert np.array_equal(solve(lookup[g]), lookup[expected])
        assert np.array_equal(g, before), 'solve mutated its input'
    # Move the seed within its tile group, then across to the disconnected group.
    moved = inp.copy()
    original = tuple(map(int, np.argwhere(moved == 4)[0]))
    moved[original], moved[8, 7] = 2, 4
    moved_out = solve(moved)
    a, b = out.copy(), moved_out.copy()
    for p in (original, (8, 7)):
        a[p] = b[p] = 0
    assert np.array_equal(a, b)
    moved[8, 7], moved[3, 17] = 2, 4
    across = solve(moved)
    assert across[2, 2] == 3 and across[2, 17] == 2
    assert not np.array_equal(across, out)


def self_test():
    """Bounded semantic, mutation, count, determinism and 42-seed checks."""
    _semantic_tests()
    fallback_rng = np.random.default_rng(2026)
    fallback = {'train': [
        _make_pair(fallback_rng, WITNESS_GEOMETRIES[0], (1, 2, 3, 4),
                   'hole_witness', True),
        _make_pair(fallback_rng, WITNESS_GEOMETRIES[-1], (1, 2, 3, 4),
                   'small_witness', True)],
        'test': [_make_pair(fallback_rng, GEOMETRIES[0], (1, 2, 3, 4), 'mixed', True)]}
    validate_generated_puzzle(fallback)
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend(_features(pair) for split in ('train', 'test') for pair in p[split])
        train_geometry = [_features(pair)['geometry'] for pair in p['train']]
        macro_max = max(max(g[:2]) for g in train_geometry)
        tile_max = max(max(g[2:]) for g in train_geometry)
        for pair in p['test']:
            g = _features(pair)['geometry']
            assert max(g[:2]) > macro_max or max(g[2:]) > tile_max, 'missing structural shift'
    signature = lambda p: tuple(_key(pair[k]) for s in ('train', 'test')
                                for pair in p[s] for k in ('input', 'output'))
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + 10 * nt + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert len({f['geometry'] for f in records}) >= 25
    assert len({f['selected_map'] for f in records}) >= 60
    assert len({f['master'] for f in records}) >= 20
    assert len({f['selected'] for f in records}) >= 10
    assert {0, 1}.issubset({f['holes'] for f in records})
    assert any(f['complete_junctions'] for f in records)
    assert any(not f['erased'] for f in records)
    assert any(f['groups'] >= 3 for f in records)
    assert any(f['geometry'][2] != f['geometry'][3] for f in records)
    assert max(e for f in records for e in f['edit_sizes']) >= 4
    for role in range(4):
        assert {f['palette'][role] for f in records} == set(range(10))
    # Bad output and missing evidence must not silently pass validation.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), geometries=len({f['geometry'] for f in records}),
                selected_shapes=len({f['selected_map'] for f in records}),
                master_bitmaps=len({f['master'] for f in records}),
                selected_counts=sorted({f['selected'] for f in records}),
                enclosed_tile_counts=sorted({f['holes'] for f in records}),
                group_counts=sorted({f['groups'] for f in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
