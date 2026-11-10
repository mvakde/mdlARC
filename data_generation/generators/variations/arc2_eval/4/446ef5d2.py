"""Translation-only picture jigsaws with a geometric external corner marker.

Pieces are connected occupied masks, not bounding rectangles. Potential edge
matches are alternatives: a bounded exhaustive search checks rectangular closure
and visible pattern continuity. No input location except the anchor is a clue.
"""
import numpy as np
from functools import lru_cache

_CORNERS = ("TL", "BR", "BL", "TR")

def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)

def _checked_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value

def _resolve_counts(seed_value, num_train, num_test):
    num_train = _checked_optional_count(num_train, "num_train", 2, 10)
    num_test = _checked_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        span = 10 - num_test
        num_train = num_test + 1 + (seed_value // 4) % span
    elif num_train is None:
        span = 10 - num_test
        num_train = num_test + 1 + seed_value % span
    elif num_test is None:
        upper = min(4, num_train - 1)
        num_test = 1 + seed_value % upper

    num_train = int(num_train)
    num_test = int(num_test)

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= train <= 10, "
            "1 <= test <= 4, and train > test"
        )
    return num_train, num_test

def _marker_cells(anchor_cell, corner):
    row, col = anchor_cell

    if corner == "TL":
        return [
            (row - 1, col - 1),
            (row - 1, col),
            (row, col - 1),
        ]
    if corner == "TR":
        return [
            (row - 1, col),
            (row - 1, col + 1),
            (row, col + 1),
        ]
    if corner == "BL":
        return [
            (row, col - 1),
            (row + 1, col - 1),
            (row + 1, col),
        ]
    return [
        (row, col + 1),
        (row + 1, col),
        (row + 1, col + 1),
    ]

def _rectangles_have_blank_gap(first, second):
    r1, c1, h1, w1 = first
    r2, c2, h2, w2 = second

    return (
        r1 + h1 < r2
        or r2 + h2 < r1
        or c1 + w1 < c2
        or c2 + w2 < c1
    )

def _choose_output_origin(
    rng,
    canvas_shape,
    target_shape,
    corner,
):
    canvas_height, canvas_width = canvas_shape
    target_height, target_width = target_shape

    row_low = 1 if corner[0] == "T" else 0
    row_high = (
        canvas_height
        - target_height
        - (1 if corner[0] == "B" else 0)
    )

    col_low = 1 if corner[1] == "L" else 0
    col_high = (
        canvas_width
        - target_width
        - (1 if corner[1] == "R" else 0)
    )

    if row_low > row_high or col_low > col_high:
        return None

    return (
        int(rng.integers(row_low, row_high + 1)),
        int(rng.integers(col_low, col_high + 1)),
    )

def _puzzle_fingerprint(puzzle):
    result = []

    for split in ("train", "test"):
        result.append(len(puzzle[split]))

        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = pair[name]
                result.extend(
                    (
                        grid.shape,
                        grid.tobytes(),
                    )
                )

    return tuple(result)

def _components(mask):
    unseen=set(map(tuple,np.argwhere(mask).tolist())); out=[]
    while unseen:
        first=min(unseen); unseen.remove(first); stack=[first]; cells=[]
        for r,c in stack:
            cells.append((r,c))
            for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if p in unseen: unseen.remove(p); stack.append(p)
        out.append(cells)
    return out

def _extract(g):
    bg=int(np.bincount(g.ravel(),minlength=10).argmax()); candidates=[]
    for col in range(10):
        q=np.argwhere(g==col)
        if col==bg or len(q)!=3: continue
        a,b=q.min(0); z,w=q.max(0)
        if z-a!=1 or w-b!=1: continue
        missing={(int(a),int(b)),(int(a),int(w)),(int(z),int(b)),(int(z),int(w))}-set(map(tuple,q.tolist()))
        p=missing.pop()
        if g[p] in (bg,col): continue
        corner={(1,1):'TL',(1,0):'TR',(0,1):'BL',(0,0):'BR'}[(p[0]-a,p[1]-b)]
        candidates.append((col,p,corner))
    assert len(candidates)==1,candidates
    marker,point,corner=candidates[0]; pieces=[]; anchor=None
    for cells in _components((g!=bg)&(g!=marker)):
        q=np.array(cells); r,c=q.min(0); h,w=q.max(0)-q.min(0)+1
        m=np.full((h,w),-1,dtype=int)
        for y,x in cells: m[y-r,x-c]=g[y,x]
        if point in cells: anchor=len(pieces)
        pieces.append((m,(int(r),int(c))))
    assert anchor is not None
    return bg,marker,point,corner,pieces,anchor



# Exact source data are regression fixtures, never generation templates.
_SOURCE_CASES = ((('88888888888888888777', '88888888877777788222', '88888888800070788777', '88888888807000788777', '87777788888888888888', '87000088888888888888', '87070088888888877788', '87777788888888822788', '87272288888888872788', '87222788888888877748', '87777788888888888448', '88888888888888888888'), ('88888888888888888888', '88888888888888888888', '88888888888888888888', '88888887777777777788', '88888887000000070788', '88888887070007000788', '88888887777777777788', '88888887272222222788', '88888887222777772788', '88888887777777777788', '88888888888888888888', '88888888888888888888')), (('8888888888', '8448889938', '8433383338', '8839983338', '8839988888', '8888888888', '8333883998', '8993883938', '8993883338', '8888888888'), ('8888888888', '8888888888', '8833333388', '8839999388', '8839999388', '8839999388', '8839333388', '8833333388', '8888888888', '8888888888')), (('44444444444444444444444444', '44444444444444444411114111', '41311444444444444413334661', '41333444444444444413114161', '41111444444444444413114161', '41111444444444444444444661', '41166411113144441171114111', '41161433333144441171114444', '44444411111144447771114444', '44444444444444441111114444', '44444444444444441111114444', '11144444444444444444444444', '66644444444444444444444444', '11144441111114444444444444', '41144443333314444444444444', '46644441111314444444111111', '41144441111314116114777111', '44444444444444116664117111', '44444444444444111114117111', '44444444444444111144444444', '44171144444444177744444444', '44177744444444171144444444', '44111144444444171144444444', '42111144444444171144444444', '42244444444444444444444444', '44444444444444444444444444'), ('44444444444444444444444444', '44444444444444444444444444', '44111111111144444444444444', '44133333333144444444444444', '44131111113144444444444444', '44131111113144444444444444', '44131111113144444444444444', '44133333333144444444444444', '44111111111144444444444444', '44111111111144444444444444', '44116666666144444444444444', '44116111116144444444444444', '44116111116144444444444444', '44116666666144444444444444', '44111111111144444444444444', '44111111111144444444444444', '44177777711144444444444444', '44171111711144444444444444', '44171111711144444444444444', '44171111711144444444444444', '44171111711144444444444444', '44177777711144444444444444', '44111111111144444444444444', '44111111111144444444444444', '44444444444444444444444444', '44444444444444444444444444')), (('88888888888888888888', '88888899998888888448', '88888833998888899948', '99988893998888893988', '93388833338888833988', '93988893338888833988', '93388833998888888888', '93988899998888888888', '93388888888889998888', '99988888888889998888', '88888888888889338888', '88888888888883398888', '88888888888883398888', '88899988888889338888', '88893988888889998888', '88833388888888888888', '88893388888888888888', '88893388888339888888', '88833388888339888888', '88899988888999888888'), ('88888888888888888888', '88888888888888888888', '88999999999999999988', '88933339999993993988', '88939939993333333988', '88933333333993333988', '88939933333993333988', '88933339993333333988', '88999999999999999988', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888', '88888888888888888888')))


class _SearchLimit(Exception):
    """An incomplete search is never accepted as a uniqueness proof."""


@lru_cache(maxsize=192)
def _packings(mask_keys, anchor, anchor_pos, point, corner, canvas):
    """Enumerate every unrotated mask tiling in every feasible area factorization.

    At the first uncovered cell a new piece must place its first occupied cell.
    Earlier cells are already occupied, so this is complete even for notches.
    Absolute scattered positions of all non-anchor pieces are deliberately absent.
    """
    masks = [np.frombuffer(b, dtype=np.uint8).reshape(s).astype(bool)
             for s, b in mask_keys]
    area = sum(int(x.sum()) for x in masks)
    n = len(masks)
    offsets = np.cumsum([0] + [int(x.sum()) for x in masks]).tolist()
    results = []
    nodes = 0
    for h in range(1, canvas[0] + 1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h - 1 if corner[0] == 'B' else 0)
        ox = point[1] - (w - 1 if corner[1] == 'R' else 0)
        if w > canvas[1] or oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(x.shape[0] > h or x.shape[1] > w for x in masks):
            continue
        configs = []
        for mask in masks:
            coords = [(int(y), int(x)) for y, x in np.argwhere(mask)]
            bit = sum(1 << (y*w+x) for y, x in coords)
            configs.append((mask.shape, coords, coords[0][1], bit))
        ar, ac = anchor_pos[0]-oy, anchor_pos[1]-ox
        ah, aw = masks[anchor].shape
        if ar < 0 or ac < 0 or ar+ah > h or ac+aw > w:
            continue
        positions = [None] * n
        positions[anchor] = (ar, ac)
        full = (1 << area) - 1
        occupied = configs[anchor][3] << (ar*w+ac)

        def visit(occ, remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 90000 or len(results) > 2500:
                raise _SearchLimit('packing enumeration budget exceeded')
            if not remaining:
                mapping = np.full((h, w), -1, dtype=np.int32)
                labels = np.full((h, w), -1, dtype=np.int16)
                for i, (r, c) in enumerate(positions):
                    for k, (y, x) in enumerate(configs[i][1]):
                        mapping[r+y, c+x] = offsets[i]+k
                        labels[r+y, c+x] = i
                assert np.all(mapping >= 0)
                right = labels[:, :-1] != labels[:, 1:]
                down = labels[:-1, :] != labels[1:, :]
                a = np.concatenate((mapping[:, :-1][right], mapping[:-1, :][down]))
                b = np.concatenate((mapping[:, 1:][right], mapping[1:, :][down]))
                absolute = tuple((r+oy, c+ox) for r, c in positions)
                results.append(((h, w), (oy, ox), absolute, mapping, a, b))
                return
            free = full ^ occ
            pos = (free & -free).bit_length()-1
            r, c = divmod(pos, w)
            for i in remaining:
                (ph, pw), coords, first, bit = configs[i]
                x = c-first
                if x < 0 or r+ph > h or x+pw > w:
                    continue
                placed = bit << (r*w+x)
                if not placed & occ:
                    positions[i] = (r, x)
                    visit(occ | placed, tuple(j for j in remaining if j != i))
        visit(occupied, tuple(i for i in range(n) if i != anchor))
    return tuple(results)


def _candidates(pieces, anchor, point, corner, canvas):
    keys = tuple((m.shape, (m >= 0).astype(np.uint8).tobytes()) for m, _ in pieces)
    return _packings(keys, anchor, pieces[anchor][1], point, corner, canvas)


def _choose_assembly(pieces, candidates, canvas, background):
    colors = np.concatenate([m[m >= 0] for m, _ in pieces])
    best_cost, best, winner, ties = None, None, None, False
    runner_up = None
    for candidate in candidates:
        shape, origin, positions, mapping, a, b = candidate
        cost = int(np.count_nonzero(colors[a] != colors[b]))
        if best_cost is None or cost <= best_cost:
            picture = colors[mapping]
            key = (shape, origin, picture.tobytes())
            if best_cost is None or cost < best_cost:
                runner_up = best_cost
                best_cost, best, winner, ties = cost, key, candidate, False
            elif key != best:
                ties = True
        elif runner_up is None or cost < runner_up:
            runner_up = cost
    assert winner is not None, 'no rectangular packing'
    assert not ties, 'distinct assemblies have equally good pattern evidence'
    shape, (r, c), positions, mapping, _, _ = winner
    output = np.full(canvas, background, dtype=np.int8)
    output[r:r+shape[0], c:c+shape[1]] = colors[mapping]
    return output, positions, best_cost, len(candidates), runner_up


def _forced_mismatches(pieces, candidates):
    """Count unequal pixel contacts that occur in every geometric packing.

    Shape can force a local seam without fixing the whole picture. Remaining
    placements must still be distinguished by their matching pattern clues.
    """
    colors=np.concatenate([m[m>=0] for m,_ in pieces])
    forced=None
    for _,_,_,_,a,b in candidates:
        unequal=colors[a]!=colors[b]
        contacts={(min(int(i),int(j)),max(int(i),int(j))) for i,j in zip(a[unequal],b[unequal])}
        forced=contacts if forced is None else forced & contacts
        if not forced:
            return 0
    return len(forced or ())


@lru_cache(maxsize=384)
def _infer_cached(shape, data):
    grid = np.frombuffer(data, dtype=np.int8).reshape(shape)
    bg, marker, point, corner, pieces, anchor = _extract(grid)
    ar, ac = pieces[anchor][1]
    ah, aw = pieces[anchor][0].shape
    assert point == (ar + (ah-1 if corner[0]=='B' else 0),
                     ac + (aw-1 if corner[1]=='R' else 0))
    candidates = _candidates(pieces, anchor, point, corner, shape)
    result = _choose_assembly(pieces, candidates, shape, bg)
    assert not result[2] or result[2] == _forced_mismatches(pieces,candidates), \
        'an unequal contact is not forced by the piece geometry'
    return result


def solve(train_pairs, test_input):
    """Prediction uses only the input and the geometric rule shown in training.

    This is an authored solver, not a protected independent reference solver.
    Source regressions also exercise nonzero but geometrically forced seams.
    """
    for pair in train_pairs:
        bg, marker, _, _, _, _ = _extract(np.asarray(pair['input']))
        assert marker not in pair['output'] and bg in pair['output']
    grid = np.asarray(test_input, dtype=np.int8)
    return _infer_cached(grid.shape, grid.tobytes())[0].copy()


def _composition(rng, total, count, minimum=2):
    parts = [minimum] * count
    for _ in range(total-count*minimum):
        parts[int(rng.integers(count))] += 1
    return parts


def _partition(rng, h, w, count, mode, stepped):
    """Compositional partitions, with optional connected complementary tabs."""
    rectangles = []
    if mode == 'grid':
        rows = 2 if count % 2 == 0 and count <= 8 else 1
        cols = count // rows
        if w < 2*cols or h < 2*rows:
            mode = 'recursive'
        else:
            y = 0
            for ph in _composition(rng, h, rows, 3 if h >= 3*rows else 2):
                x = 0
                for pw in _composition(rng, w, cols, 3 if w >= 3*cols else 2):
                    rectangles.append((y, x, ph, pw)); x += pw
                y += ph
    if mode in ('bands', 'outer'):
        bands = 2 if mode == 'outer' else min(int(rng.integers(2, 5)), count, w//2)
        counts = [1] * bands
        for _ in range(count-bands):
            choices = [i for i in range(bands) if counts[i] < h//2 and (mode!='outer' or i!=0)]
            if not choices:
                mode = 'recursive'; break
            counts[int(rng.choice(choices))] += 1
        if mode != 'recursive':
            x = 0
            for pw, n in zip(_composition(rng, w, bands), counts):
                y = 0
                for ph in _composition(rng, h, n):
                    rectangles.append((y, x, ph, pw)); y += ph
                x += pw
    if mode == 'recursive':
        rectangles = [(0, 0, h, w)]
        for _ in range(count-1):
            choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rectangles)
                       for axis, length in enumerate((ph, pw)) if length >= 4]
            if not choices:
                return None
            weights = np.array([rectangles[i][2]*rectangles[i][3] for i, _ in choices], float)
            i, axis = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            r, c, ph, pw = rectangles.pop(i)
            cut = int(rng.integers(2, (ph if axis == 0 else pw)-1))
            if axis == 0:
                rectangles.extend([(r, c, cut, pw), (r+cut, c, ph-cut, pw)])
            else:
                rectangles.extend([(r, c, ph, cut), (r, c+cut, ph, pw-cut)])
    labels = np.full((h, w), -1, dtype=int)
    for i, (r, c, ph, pw) in enumerate(rectangles):
        labels[r:r+ph, c:c+pw] = i
    if np.any(labels < 0):
        return None
    if stepped:
        options = []
        for i, (r, c, ph, pw) in enumerate(rectangles):
            if ph >= 3 and pw >= 3:
                for y in range(r+1, r+ph-1):
                    for x, dx in ((c, -1), (c+pw-1, 1)):
                        if 0 <= x+dx < w and labels[y, x+dx] != i:
                            options.append((y, x, int(labels[y, x+dx])))
                for x in range(c+1, c+pw-1):
                    for y, dy in ((r, -1), (r+ph-1, 1)):
                        if 0 <= y+dy < h and labels[y+dy, x] != i:
                            options.append((y, x, int(labels[y+dy, x])))
        if not options:
            return None
        for k in rng.permutation(len(options))[:int(rng.integers(1, 4))]:
            y, x, recipient = options[int(k)]
            donor = int(labels[y, x]); before = labels.copy()
            # Extend the tab along a seam, sometimes to an endpoint. This produces
            # true steps as well as interior notches, with 1–2 cell depths and
            # 1–4 cell runs. Connectivity is checked on occupied masks.
            dy, dx = next(((dy,dx) for dy,dx in ((-1,0),(1,0),(0,-1),(0,1))
                           if 0<=y+dy<h and 0<=x+dx<w
                           and labels[y+dy,x+dx]==recipient), (0,0))
            length, depth = int(rng.integers(1,5)), int(rng.integers(1,3))
            sign = 1 if rng.integers(2) else -1
            for along in range(length):
                yy,xx=y+along*dx*sign,x+along*dy*sign
                if not (0<=yy+dy<h and 0<=xx+dx<w
                        and 0<=yy<h and 0<=xx<w
                        and before[yy+dy,xx+dx]==recipient):
                    break
                for inward in range(depth):
                    ry,rx=yy-inward*dy,xx-inward*dx
                    if 0<=ry<h and 0<=rx<w and before[ry,rx]==donor:
                        labels[ry,rx]=recipient
            if any(len(_components(labels == i)) != 1 for i in (donor, recipient)):
                labels = before
        if all(np.sum(labels==i) == np.prod(np.ptp(np.argwhere(labels==i), axis=0)+1)
               for i in range(len(rectangles))):
            return None
    if rng.integers(2):
        labels = labels.T.copy()
    if rng.integers(2):
        labels = labels[::-1].copy()
    if rng.integers(2):
        labels = labels[:, ::-1].copy()
    return labels


def _picture(rng, labels, palette, kind, relaxed, irregular=False):
    """Draw a whole coherent picture before cutting; smooth selected cut contacts.

    Equality groups make repeated/quiet edge evidence possible, without requiring
    candidate edge matches to be unique or all contacts to be informative.
    """
    h, w = labels.shape
    carrier, patterns = palette[0], palette[1:]
    pic = np.full((h, w), carrier, dtype=np.int8)
    if kind == 'regions':
        for k in range(int(rng.integers(3, 8))):
            y, x = int(rng.integers(h-1)), int(rng.integers(w-1))
            ph, pw = int(rng.integers(2, h-y+1)), int(rng.integers(2, w-x+1))
            pic[y:y+ph, x:x+pw] = patterns[k % len(patterns)]
    elif kind == 'outlines':
        for k in range(int(rng.integers(1, 5))):
            y, x = int(rng.integers(max(1,h-3))), int(rng.integers(max(1,w-3)))
            ph, pw = int(rng.integers(3,h-y+1)), int(rng.integers(3,w-x+1))
            col = patterns[k % len(patterns)]
            pic[y:y+ph, x] = col; pic[y:y+ph, x+pw-1] = col
            pic[y, x:x+pw] = col; pic[y+ph-1, x:x+pw] = col
    elif kind == 'paths':
        for k in range(int(rng.integers(2, 6))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(h+w, 3*(h+w)))):
                pic[y:y+int(rng.integers(1,3)), x:x+int(rng.integers(1,3))] = patterns[k % len(patterns)]
                dy, dx = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
                y, x = min(h-1,max(0,y+dy)), min(w-1,max(0,x+dx))
    else:
        # Repeated blocks or bars with independently varied lengths and accents.
        for y in range(int(rng.integers(2)), h, int(rng.integers(3, 6))):
            for x in range(int(rng.integers(2)), w, int(rng.integers(3, 6))):
                pic[y:y+2, x:x+int(rng.integers(2,5))] = patterns[(y+x) % len(patterns)]
    if rng.random() < .6:
        pic[0,:] = carrier; pic[-1,:] = carrier
        pic[:,0] = carrier; pic[:,-1] = carrier
    parent = list(range(h*w))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for y in range(h):
        for x in range(w):
            for yy, xx in ((y+1,x),(y,x+1)):
                if yy < h and xx < w and labels[y,x] != labels[yy,xx]:
                    if not relaxed or rng.random() > .32:
                        a, b = root(y*w+x), root(yy*w+xx)
                        parent[b] = a
    groups = {}
    for i in range(h*w):
        groups.setdefault(root(i), []).append(i)
    flat = pic.ravel()
    for cells in groups.values():
        values, counts = np.unique(flat[cells], return_counts=True)
        if irregular and len(cells) > 1:
            value = int(rng.choice(palette))
        else:
            value = int(rng.choice(values[counts == counts.max()]))
        flat[cells] = value
    # Every payload color is visible, and cannot masquerade as the 3-cell marker.
    for color in palette:
        for _ in range(12):
            if np.count_nonzero(pic == color) >= 4:
                break
            y,x=int(rng.integers(h-1)),int(rng.integers(w-1))
            for yy,xx in ((y,x),(y+1,x),(y,x+1),(y+1,x+1)):
                flat[groups[root(yy*w+xx)]] = color
    if any(np.count_nonzero(pic == c) < 4 for c in palette):
        return None
    return pic


def _pieces_from_picture(labels, pic):
    pieces = []
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i); r, c = q.min(0); y, x = q.max(0)+1
        matrix = np.where(labels[r:y,c:x] == i, pic[r:y,c:x], -1)
        pieces.append((matrix.astype(np.int8), (int(r),int(c))))
    return pieces


def _scatter(rng, pieces, anchor, final, marker_cells, canvas, stationary):
    remaining = [i for i in range(len(pieces)) if i != anchor]
    remaining.sort(key=lambda i: -pieces[i][0].size)
    for _ in range(14):
        positions = {anchor: final[anchor]}
        rects = [(final[anchor][0],final[anchor][1],*pieces[anchor][0].shape)]
        if stationary:
            eligible = [i for i in remaining if _rectangles_have_blank_gap(
                (*final[i], *pieces[i][0].shape), rects[0])]
            if eligible:
                i = int(rng.choice(eligible)); positions[i] = final[i]
                rects.append((*final[i], *pieces[i][0].shape))
        failed = False
        for i in remaining:
            if i in positions:
                continue
            h, w = pieces[i][0].shape
            options = [(r,c) for r in range(canvas[0]-h+1) for c in range(canvas[1]-w+1)]
            found = False
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                if any(r-1 <= y <= r+h and c-1 <= x <= c+w for y,x in marker_cells):
                    continue
                if all(_rectangles_have_blank_gap((r,c,h,w), rect) for rect in rects):
                    positions[i] = (r,c); rects.append((r,c,h,w)); found = True; break
            if not found:
                failed = True; break
        if not failed:
            return [positions[i] for i in range(len(pieces))]
    return None


def _generate_pair(rng, corner, roles, size, stepped, relaxed, stationary, mode):
    bg, marker = roles
    palette_size = int(rng.integers(2, 7))
    available = [c for c in range(10) if c not in roles]
    palette = [int(c) for c in rng.permutation(available)[:palette_size]]
    kind = str(rng.choice(('regions', 'outlines', 'paths', 'repeated')))
    for attempt in range(28):
        if size == 'small':
            h, w = int(rng.integers(6,10)), int(rng.integers(6,11))
            count = int(rng.integers(3,6))
            if mode == 'grid' and rng.random() < .5:
                h, w, count = 6, 6, 4
        elif size == 'large':
            h, w = int(rng.integers(6,12)), int(rng.integers(15,23))
            count = int(rng.integers(6,11))
        else:
            h, w = int(rng.integers(6,14)), int(rng.integers(8,17))
            count = int(rng.integers(3,9))
        labels = _partition(rng, h, w, count, mode, stepped)
        if labels is None:
            continue
        h, w = labels.shape
        area = h*w
        dummy = np.zeros_like(labels)
        pieces = _pieces_from_picture(labels, dummy)
        point_local = (h-1 if corner[0]=='B' else 0, w-1 if corner[1]=='R' else 0)
        anchor = int(labels[point_local])
        base_h = max(h+2, int(np.ceil(np.sqrt(area*2.2))))
        base_w = max(w+2, int(np.ceil(area*2.2/base_h)))
        canvas = (min(30,base_h+int(rng.integers(0,7))), min(30,base_w+int(rng.integers(0,7))))
        compact = (h,w)==(6,6) and count==4 and rng.random()<.4
        if compact:
            canvas=(10,10)
        elif h<=9 and w<=17 and area<=120 and rng.random()<.12:
            canvas=(12,20)
        origin = _choose_output_origin(rng, canvas, (h,w), corner)
        if compact:
            origin=(2,2)
        point = (origin[0]+point_local[0],origin[1]+point_local[1])
        markers = _marker_cells(point, corner)
        final = [(origin[0]+r,origin[1]+c) for _,(r,c) in pieces]
        positions = _scatter(rng, pieces, anchor, final, markers, canvas, stationary)
        if positions is None:
            continue
        scattered = [(m,p) for (m,_),p in zip(pieces,positions)]
        try:
            candidates = _candidates(scattered, anchor, point, corner, canvas)
        except _SearchLimit:
            continue
        if not candidates:
            continue
        for picture_attempt in range(16):
            pic = _picture(rng, labels, palette, kind, relaxed,
                           irregular=picture_attempt >= 12)
            if pic is None:
                continue
            colored = _pieces_from_picture(labels, pic)
            scattered = [(m,p) for (m,_),p in zip(colored,positions)]
            try:
                output, recovered, cost, alternatives, _ = _choose_assembly(scattered,candidates,canvas,bg)
            except AssertionError:
                continue
            # No numerical mismatch tolerance: only geometrically unavoidable
            # contacts may disagree; every remaining contact must match.
            if cost and cost != _forced_mismatches(scattered,candidates):
                continue
            expected = np.full(canvas,bg,dtype=np.int8)
            expected[origin[0]:origin[0]+h,origin[1]:origin[1]+w] = pic
            if not np.array_equal(output, expected):
                continue
            grid = np.full(canvas,bg,dtype=np.int8)
            for (m,_),(r,c) in zip(colored,positions):
                sub = grid[r:r+m.shape[0],c:c+m.shape[1]]
                sub[m>=0] = m[m>=0]
            for p in markers:
                grid[p] = marker
            if int(np.bincount(grid.ravel(),minlength=10).argmax()) != bg:
                continue
            try:
                _extract(grid)
            except AssertionError:
                continue
            return {'input':grid,'output':output}
    return _fallback_pair(rng,corner,roles,palette)


def _fallback_pair(rng, corner, roles, palette):
    """Guaranteed geometry-only closure, with no stochastic failure loop.

    A full-height strip and two unequal-height wide pieces have just one feasible
    packing when a wide piece is anchored. The 27-cell area cannot form another
    aspect ratio accommodating the 7-cell-wide pieces. Reflect before cutting.
    """
    labels = np.zeros((3,9),dtype=int)
    labels[0,2:] = 1; labels[1:,2:] = 2
    if corner[1] == 'L':
        labels = labels[:,::-1].copy()
    pic = np.full(labels.shape,palette[0],dtype=np.int8)
    flat = pic.ravel()
    for i,color in enumerate(palette):
        flat[4*i:4*i+4] = color
    pieces = _pieces_from_picture(labels,pic)
    canvas = (int(rng.integers(20,26)),int(rng.integers(22,28)))
    origin = (int(rng.integers(5,8)),int(rng.integers(6,10)))
    local = (2 if corner[0]=='B' else 0,8 if corner[1]=='R' else 0)
    point = (origin[0]+local[0],origin[1]+local[1])
    anchor = int(labels[local])
    final = [(origin[0]+r,origin[1]+c) for _,(r,c) in pieces]
    positions = [None]*3
    positions[anchor] = final[anchor]
    for i,p in zip((i for i in range(3) if i!=anchor),((1,1),(14,12))):
        positions[i] = p
    grid = np.full(canvas,roles[0],dtype=np.int8)
    out = grid.copy();out[origin[0]:origin[0]+3,origin[1]:origin[1]+9]=pic
    for (m,_),(r,c) in zip(pieces,positions):
        grid[r:r+m.shape[0],c:c+m.shape[1]]=m
    for p in _marker_cells(point,corner):
        grid[p]=roles[1]
    return {'input':grid,'output':out}


def _edge_segments(matrix):
    """Every contiguous straight exposed boundary, including notch walls."""
    mask = matrix >= 0
    h, w = mask.shape
    segments = []
    for name, dy, dx in (('T',-1,0),('B',1,0),('L',0,-1),('R',0,1)):
        lines = {}
        for r,c in np.argwhere(mask):
            r,c = int(r),int(c)
            y,x = r+dy,c+dx
            if 0 <= y < h and 0 <= x < w and mask[y,x]:
                continue
            fixed, along = (r,c) if dy else (c,r)
            lines.setdefault(fixed,[]).append((along,int(matrix[r,c])))
        for fixed, cells in lines.items():
            cells.sort(); run=[]; start=None; previous=None
            for along, value in cells:
                if previous is not None and along != previous+1:
                    segments.append((name,fixed,start,tuple(run)));run=[]
                if not run:
                    start=along
                run.append(value);previous=along
            segments.append((name,fixed,start,tuple(run)))
    return segments


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    bg, marker, point, corner, pieces, anchor = _extract(input_grid)
    predicted, positions, cost, alternatives, runner_up = _infer_cached(
        input_grid.shape,input_grid.astype(np.int8).tobytes())
    assert np.array_equal(predicted,output_grid), 'output is not the input-derived unique assembly'
    assert positions[anchor] == pieces[anchor][1], 'anchor moved'
    assert not np.any(output_grid == marker)
    assert all(output_grid[p] == bg for p in _marker_cells(point,corner))
    assert not np.array_equal(input_grid,output_grid)
    cin = np.bincount(input_grid.ravel(),minlength=10)
    cout = np.bincount(output_grid.ravel(),minlength=10)
    assert cin[marker] == 3 and cout[bg] == cin[bg]+3
    assert all(cin[c] == cout[c] for c in range(10) if c not in (bg,marker))
    occupied = np.zeros_like(output_grid,dtype=bool)
    labels = np.full(output_grid.shape,-1,dtype=int)
    stepped = 0; stationary = 0; small = 0; colored_corners = 0
    payload = tuple(int(c) for c in np.unique(output_grid) if c != bg)
    carrier = max(payload,key=lambda c: int(cout[c]))
    edges=[]; dimensions=[]; piece_edges=[]
    for i, ((matrix,old),(r,c)) in enumerate(zip(pieces,positions)):
        h,w=matrix.shape;mask=matrix>=0
        assert len(_components(mask)) == 1
        assert not occupied[r:r+h,c:c+w][mask].any()
        occupied[r:r+h,c:c+w][mask]=True
        labels[r:r+h,c:c+w][mask]=i
        assert np.array_equal(output_grid[r:r+h,c:c+w][mask],matrix[mask])
        stepped += int(mask.sum() < h*w)
        small += int(min(h,w) <= 3)
        stationary += int(i != anchor and old == (r,c))
        for p in ((0,0),(0,w-1),(h-1,0),(h-1,w-1)):
            colored_corners += int(matrix[p] >= 0 and matrix[p] != carrier)
        segments=_edge_segments(matrix)
        edges.extend(segments);piece_edges.append(segments);dimensions.append((h,w))
    q=np.argwhere(occupied);r,c=q.min(0);y,x=q.max(0)+1
    target=(int(y-r),int(x-c))
    assert occupied[r:y,c:x].all()
    assert int(occupied.sum()) == target[0]*target[1]
    expected_point=(int(y-1 if corner[0]=='B' else r),int(x-1 if corner[1]=='R' else c))
    assert point==expected_point
    weak=sum(int(len(seq)>=2 and all(v==carrier for v in seq)) for _,_,_,seq in edges)
    signatures=[seq for _,_,_,seq in edges if len(seq)>=2]
    repeats=len(signatures)-len(set(signatures))
    seam_groups={}
    for rr in range(int(r),int(y)):
        for cc in range(int(c),int(x)):
            for yy,xx in ((rr+1,cc),(rr,cc+1)):
                if yy>=y or xx>=x:
                    continue
                a,b=int(labels[rr,cc]),int(labels[yy,xx])
                if a != b:
                    seam_groups.setdefault((a,b,'D' if yy!=rr else 'R'),[]).append(
                        (int(output_grid[rr,cc]),int(output_grid[yy,xx])))
    unequal=sum(any(a!=b for a,b in cells) for cells in seam_groups.values())
    subedges=0
    for (a,b,direction),cells in seam_groups.items():
        da=dimensions[a][1 if direction=='D' else 0]
        db=dimensions[b][1 if direction=='D' else 0]
        subedges+=int(len(cells) < max(da,db))
    false_matches=0
    for i,first in enumerate(piece_edges):
        for j,second in enumerate(piece_edges):
            if i==j:
                continue
            for side,fixed,start,seq in first:
                if side not in ('R','B') or len(seq)<2:
                    continue
                for facing,other_fixed,other_start,other in second:
                    if facing != ('L' if side=='R' else 'T') or len(other)<2:
                        continue
                    for offset in range(min(0,len(seq)-len(other)),max(0,len(seq)-len(other))+1):
                        a,b=max(0,offset),max(0,-offset)
                        length=min(len(seq)-a,len(other)-b)
                        if seq[a:a+length]!=other[b:b+length]:
                            continue
                        across=fixed+1-other_fixed
                        along=start+offset-other_start
                        delta=(along,across) if side=='R' else (across,along)
                        real=(positions[j][0]-positions[i][0],positions[j][1]-positions[i][1])
                        false_matches+=int(delta!=real)
    # A color-normalized output mask partition measures structure, not RNG metadata.
    crop=labels[r:y,c:x]
    vertical_cuts=sum(bool(np.all(crop[:,i]!=crop[:,i-1])) for i in range(1,crop.shape[1]))
    horizontal_cuts=sum(bool(np.all(crop[i,:]!=crop[i-1,:])) for i in range(1,crop.shape[0]))
    band_subdivisions=[]
    for array in (crop,crop.T):
        cuts=[0]+[i for i in range(1,array.shape[1]) if np.all(array[:,i]!=array[:,i-1])]+[array.shape[1]]
        band_subdivisions.append(tuple(len(np.unique(array[:,a:b])) for a,b in zip(cuts,cuts[1:])))
    outer_subdivided=(any(ph==target[0] and pw<target[1] for ph,pw in dimensions)
                      and any(ph<target[0] for ph,pw in dimensions)) or (
                      any(pw==target[1] and ph<target[0] for ph,pw in dimensions)
                      and any(pw<target[1] for ph,pw in dimensions))
    picture=output_grid[r:y,c:x]
    motif_regions=[len(q) for color in payload if color!=carrier
                   for q in _components(picture==color)]
    rename={};normalized=[]
    for v in crop.ravel():
        v=int(v);rename.setdefault(v,len(rename));normalized.append(rename[v])
    forced_cost=0 if not cost else _forced_mismatches(
        pieces,_candidates(pieces,anchor,point,corner,input_grid.shape))
    return dict(background=bg,marker=marker,carrier=carrier,payload_colors=payload,
                corner=corner,piece_count=len(pieces),target_shape=target,
                canvas_shape=input_grid.shape,target_area=target[0]*target[1],
                stepped_pieces=stepped,stationary_nonanchors=stationary,
                small_pieces=small,colored_corners=colored_corners,
                quiet_edges=weak,repeated_signatures=repeats,unequal_contacts=unequal,
                false_edge_matches=false_matches,
                subedge_contacts=subedges,packing_count=alternatives,mismatch_cells=cost,
                forced_mismatch_cells=forced_cost,
                vertical_cuts=vertical_cuts,horizontal_cuts=horizontal_cuts,
                full_height_piece=any(ph==target[0] and pw<target[1] for ph,pw in dimensions),
                full_width_piece=any(pw==target[1] and ph<target[0] for ph,pw in dimensions),
                largest_pattern_region=max(motif_regions,default=0),
                four_three_by_three=(dimensions==[(3,3)]*4),
                outer_subdivided=outer_subdivided,
                mixed_bands=any(len(bands)>=3 and min(bands)==1 and max(bands)>1
                                for bands in band_subdivisions),
                notch_area=sum(int(np.sum(m<0)) for m,_ in pieces),
                margin=None if runner_up is None else runner_up-cost,
                piece_dimensions=tuple(dimensions),
                layout=(target,tuple(normalized)))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value=_normalize_seed(seed)
    nt,ne=_resolve_counts(seed_value,num_train,num_test)
    rng=np.random.default_rng(seed_value)
    # These are sampling regimes, not hidden solution conventions. The only shared
    # interpretation is translation plus external geometric anchoring.
    regime=int(rng.integers(5))
    if regime==0:
        train_corners=('TL','BR');test_corners=('BL','TR')
    elif regime==1:
        train_corners=('BL','TR');test_corners=('TL','BR')
    elif regime==2:
        train_corners=tuple(str(c) for c in rng.choice(_CORNERS,2,replace=False))
        test_corners=_CORNERS
    elif regime==3:
        train_corners=(str(rng.choice(_CORNERS)),);test_corners=train_corners
    else:
        train_corners=tuple(str(c) for c in rng.choice(_CORNERS,2,replace=False))
        test_corners=train_corners
    broad=bool(rng.random()<.75)
    scale_shift=bool(rng.random()<.4)
    base=tuple(int(c) for c in rng.choice(10,2,replace=False))
    varied_roles=bool(rng.random()<.85)
    modes=('grid','outer','bands','recursive')
    train_modes=[str(rng.choice(modes)) for _ in range(nt)]
    if train_modes[0]==train_modes[1]:
        train_modes[1]=modes[(modes.index(train_modes[0])+int(rng.integers(1,4)))%4]
    train_plan=[]
    corner_plan=list(train_corners)+[str(rng.choice(train_corners)) for _ in range(nt-len(train_corners))]
    rng.shuffle(corner_plan)
    broad_index=int(rng.integers(nt))
    for i in range(nt):
        role_kind=int(rng.integers(3)) if varied_roles else 0
        if varied_roles and i==1:
            role_kind=int(rng.integers(1,3))
        roles=base if role_kind==0 else base[::-1] if role_kind==1 else tuple(int(c) for c in rng.choice(10,2,replace=False))
        size='small' if i==0 else 'medium' if scale_shift else str(rng.choice(('small','medium','large'),p=(.25,.55,.2)))
        steps=broad and (i==broad_index or rng.random()<.45)
        relaxed=(i==broad_index or rng.random()<.35)
        train_plan.append((corner_plan[i],roles,size,steps,relaxed,rng.random()<.35,train_modes[i]))
    # Demonstration order does not carry a count, direction or size convention.
    rng.shuffle(train_plan)
    puzzle={'train':[],'test':[]}
    for plan in train_plan:
        puzzle['train'].append(_generate_pair(rng,*plan))
    has_steps=any(_analyze_pair(p['input'],p['output'])['stepped_pieces'] for p in puzzle['train'])
    seen_inputs={(p['input'].shape,p['input'].tobytes()) for p in puzzle['train']}
    for i in range(ne):
        corner=str(rng.choice(test_corners))
        role_kind=int(rng.integers(3)) if varied_roles else 0
        roles=base if role_kind==0 else base[::-1] if role_kind==1 else tuple(int(c) for c in rng.choice(10,2,replace=False))
        size='large' if scale_shift else str(rng.choice(('small','medium','large'),p=(.2,.45,.35)))
        plan=(corner,roles,size,has_steps and rng.random()<.65,rng.random()<.4,
              rng.random()<.35,str(rng.choice(modes)))
        pair=_generate_pair(rng,*plan)
        key=(pair['input'].shape,pair['input'].tobytes())
        if key in seen_inputs:
            # Deterministic fresh configuration, without changing the interpretation.
            pair=_generate_pair(rng,*plan)
        puzzle['test'].append(pair)
        seen_inputs.add((pair['input'].shape,pair['input'].tobytes()))
    validate_generated_puzzle(puzzle)
    return puzzle


def _rows(grid):
    return tuple(''.join(str(int(c)) for c in row) for row in grid)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    inputs=set();infos={'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            for g in pair.values():
                assert isinstance(g,np.ndarray) and g.ndim==2
                assert np.issubdtype(g.dtype,np.integer)
                assert all(1<=d<=30 for d in g.shape)
                assert int(g.min())>=0 and int(g.max())<=9
            key=(pair['input'].shape,pair['input'].astype(np.int8).tobytes())
            assert key not in inputs;inputs.add(key)
            assert (_rows(pair['input']),_rows(pair['output'])) not in _SOURCE_CASES
            try:
                info=_analyze_pair(pair['input'],pair['output'])
            except _SearchLimit as exc:
                raise AssertionError('uniqueness search did not finish') from exc
            assert 3<=info['piece_count']<=10
            assert 2<=len(info['payload_colors'])<=6
            # Geometry may force local unequal seams while other placements still
            # need pattern evidence. No optional unequal seam is accepted.
            assert info['mismatch_cells']==info['forced_mismatch_cells']
            infos[split].append(info)
    train=infos['train'];test=infos['test']
    tc={x['corner'] for x in train}
    if len(tc)==1:
        assert {x['corner'] for x in test} <= tc
    if any(x['stepped_pieces'] for x in test):
        assert any(x['stepped_pieces'] for x in train)
    for pair in puzzle['test']:
        prediction=solve([{k:g.copy() for k,g in p.items()} for p in puzzle['train']],pair['input'].copy())
        assert np.array_equal(prediction,pair['output'])


def _source_regressions():
    infos=[]
    for input_rows,output_rows in _SOURCE_CASES:
        grid=np.array([[int(c) for c in row] for row in input_rows],dtype=np.int8)
        expected=np.array([[int(c) for c in row] for row in output_rows],dtype=np.int8)
        # Prediction receives no expected label or source index.
        assert np.array_equal(solve([],grid.copy()),expected)
        infos.append(_analyze_pair(grid,expected))
    assert infos[0]['unequal_contacts']==2
    assert infos[2]['stepped_pieces']==2 and infos[2]['stationary_nonanchors']==1
    assert len(infos[1]['payload_colors'])==len(infos[3]['payload_colors'])==2


def self_test() -> None:
    _source_regressions()
    for corner in _CORNERS:
        for payload_count in (2,6):
            pair=_fallback_pair(np.random.default_rng(71),corner,(0,1),list(range(2,2+payload_count)))
            assert _analyze_pair(pair['input'],pair['output'])['packing_count']==1
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    train_counts={len(p['train']) for p in puzzles};test_counts={len(p['test']) for p in puzzles}
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len({_puzzle_fingerprint(p) for p in puzzles})==42
    all_infos=[];within=[]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos=[_analyze_pair(p['input'],p['output']) for s in ('train','test') for p in puzzle[s]]
        all_infos.extend(infos)
        within.append({name:len({str(p[name]) for p in infos}) for name in
                       ('target_shape','piece_count','layout','payload_colors','background','marker')})
    assert len({x['canvas_shape'] for x in all_infos})>=40
    assert len({x['target_shape'] for x in all_infos})>=50
    assert len({x['layout'] for x in all_infos})>=120
    assert {x['corner'] for x in all_infos}==set(_CORNERS)
    assert {len(x['payload_colors']) for x in all_infos}==set(range(2,7))
    assert len({x['piece_count'] for x in all_infos})>=7
    assert sum(x['stepped_pieces']>0 for x in all_infos)>=25
    assert sum(x['unequal_contacts']>0 for x in all_infos)>=8
    assert sum(x['unequal_contacts']>0 and x['packing_count']>1 for x in all_infos)>=5
    assert sum(x['stationary_nonanchors']>0 for x in all_infos)>=10
    assert sum(x['quiet_edges']>0 for x in all_infos)>=80
    assert sum(x['colored_corners']>0 for x in all_infos)>=80
    assert sum(x['repeated_signatures']>0 for x in all_infos)>=80
    assert sum(x['subedge_contacts']>0 for x in all_infos)>=80
    assert sum(x['false_edge_matches']>0 for x in all_infos)>=80
    assert sum(x['four_three_by_three'] for x in all_infos)>=2
    assert sum(x['outer_subdivided'] for x in all_infos)>=20
    assert sum(x['mixed_bands'] for x in all_infos)>=10
    assert sum(x['notch_area']>=6 for x in all_infos)>=10
    assert sum(x['largest_pattern_region']>=10 for x in all_infos)>=80
    assert sum(max(x['target_shape'])/min(x['target_shape'])>=2.5 for x in all_infos)>=12
    assert sum(max(x['canvas_shape'])<=16 for x in all_infos)>=5
    assert sum(x['stepped_pieces']>0 and len(x['payload_colors'])>=4 and x['piece_count']>=6
               and x['repeated_signatures']>0 for x in all_infos)>=10
    assert all(x['layout']>=2 for x in within)
    assert sum(x['piece_count']>=2 for x in within)>=36
    assert sum(x['background']>=2 and x['marker']>=2 for x in within)>=25
    assert all(any(x['corner']==c and x['target_shape'][0]>x['target_shape'][1]
                   for x in all_infos) and
               any(x['corner']==c and x['target_shape'][0]<x['target_shape'][1]
                   for x in all_infos) for c in _CORNERS)
    assert len({(len(x['payload_colors']),x['piece_count']) for x in all_infos})>=30
    observed_regimes=set();role_changes={'train':set(),'test':set()}
    scale_increases=0;comparable=0
    for puzzle in puzzles:
        train=[_analyze_pair(p['input'],p['output']) for p in puzzle['train']]
        test=[_analyze_pair(p['input'],p['output']) for p in puzzle['test']]
        a={x['corner'] for x in train};b={x['corner'] for x in test}
        observed_regimes.add('same' if len(a|b)==1 else 'overlap' if a&b else
                             'original' if a=={'TL','BR'} else
                             'reverse' if a=={'BL','TR'} else 'disjoint')
        base=(train[0]['background'],train[0]['marker'])
        for split,values in (('train',train),('test',test)):
            for x in values:
                roles=(x['background'],x['marker'])
                role_changes[split].add('same' if roles==base else 'swap' if roles==base[::-1] else 'new')
        scale_increases+=min(x['target_area'] for x in test)>max(x['target_area'] for x in train)
        comparable+=max(x['target_area'] for x in train+test)<=3*min(x['target_area'] for x in train+test)
    assert observed_regimes >= {'same','overlap','original','reverse'}
    assert all(changes=={'same','swap','new'} for changes in role_changes.values())
    assert scale_increases>=3 and comparable>=3
    # Search and tie decisions cannot depend on segmentation order.
    pair=puzzles[1]['test'][0]
    bg,marker,point,corner,pieces,anchor=_extract(pair['input'])
    order=list(reversed(range(len(pieces))))
    reordered=[pieces[i] for i in order];anchor=order.index(anchor)
    predicted=_choose_assembly(reordered,_candidates(reordered,anchor,point,corner,pair['input'].shape),
                               pair['input'].shape,bg)[0]
    assert np.array_equal(predicted,pair['output'])
    # A plain four-piece picture permits different rectangles; reject the tie.
    pieces=[(np.full((3,3),2,dtype=np.int8),(1,1)) for _ in range(4)]
    try:
        _choose_assembly(pieces,_candidates(pieces,0,(1,1),'TL',(15,15)),(15,15),0)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous packing accepted')
    assert _puzzle_fingerprint(generate_puzzle(123456,7,3))==_puzzle_fingerprint(generate_puzzle(123456,7,3))
    for i,(nt,ne) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p=generate_puzzle(8100+i,nt,ne);validate_generated_puzzle(p)
        assert nt is None or len(p['train'])==nt
        assert ne is None or len(p['test'])==ne
    for nt,ne in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(2,1.5)):
        try:
            generate_puzzle(99,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject changed colors, an unerased marker, moved anchoring, and duplicates.
    p=puzzles[0]
    for corruption in ('color','marker','anchor','duplicate'):
        damaged={s:[{k:g.copy() for k,g in pair.items()} for pair in p[s]] for s in p}
        pair=damaged['test'][0];grid=pair['output']
        bg,marker,point,corner,_,_=_extract(pair['input'])
        if corruption=='color':
            r,c=np.argwhere(grid!=bg)[0];grid[r,c]=(int(grid[r,c])+1)%10
        elif corruption=='marker':
            grid[_marker_cells(point,corner)[0]]=marker
        elif corruption=='anchor':
            pair['output']=np.roll(grid,1,axis=0)
        else:
            damaged['train'][1]={k:g.copy() for k,g in damaged['train'][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: '+corruption)
