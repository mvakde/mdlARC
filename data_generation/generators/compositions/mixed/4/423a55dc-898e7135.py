"""Rotation-only blueprint assembly followed by pixel-row, bottom-anchored shear.

Only NumPy and the standard library are used. No parent code or saved labels are
loaded. Recovery locates and validates the blueprint and scale from the input.
"""
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 819617090, 65771479, 853426780, 689886786, 385075487,
              924964993, 906220587, 171294157, 547805843, 725475176,
              569475560, 357562543, 45103536, 797762629, 433635995,
              699144929, 10514111, 138906611, 900360860, 570879478,
              219927468, 254343119, 759205327, 677590186, 472008776,
              552725120, 90534435, 888740503, 708558988, 271433771,
              825567858, 183742207, 253972316, 928117448, 370670815,
              389897478, 837949588, 852984063, 122836911, 971677848,
              49794857)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _turn(shape):
    return _normal([(c, -r) for r, c in shape])


def _rotations(shape):
    shape = _normal(shape)
    forms = {shape}
    for _ in range(3):
        shape = _turn(shape)
        forms.add(shape)
    return tuple(sorted(forms))


def _canonical(shape):
    return _rotations(shape)[0]


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        first = min(pending)
        pending.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                p = r+dr, c+dc
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _mask(shape):
    a = np.zeros(_size(shape), dtype=bool)
    for r, c in shape:
        a[r, c] = True
    return a


def _expand(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def _shear(board, background):
    """Shear physical rows, using the complete board's bottom, on its crop."""
    h, w = board.shape
    out = np.full_like(board, background)
    for r in range(h):
        distance = h-1-r
        if distance < w:
            out[r, :w-distance] = board[r, distance:]
    return out


def _decode(cells, k):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    blocks = Counter(((r-r0)//k, (c-c0)//k) for r, c in cells)
    if any(n != k*k for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _visible_pixels(cells, k, board_height):
    total = 0
    for r, c in cells:
        for dr in range(k):
            shift = board_height-1-(r*k+dr)
            total += k-min(k, max(0, shift-c*k))
    return total


def _proximity_conflict(hg, pg, bounds, answer):
    """An equal-area pair whose shape-correct assignment costs more distance.

    Squared centroid distances use the actual input coordinates, not output
    positions. Strict inequality excludes a positional tie as the explanation.
    """
    def center(cells):
        return np.mean(np.asarray(cells, dtype=float), axis=0)

    classes = sorted(hg)
    for i, s in enumerate(classes):
        for t in classes[i+1:]:
            if (len(s) != len(t) or len(hg[s]) != 1 or len(hg[t]) != 1
                    or pg[s][0][0] == pg[t][0][0]
                    or not np.any(answer == pg[s][0][0])
                    or not np.any(answer == pg[t][0][0])):
                continue
            hs = center(hg[s][0]) + [bounds[0], bounds[2]]
            ht = center(hg[t][0]) + [bounds[0], bounds[2]]
            ps, pt = center(pg[s][0][2]), center(pg[t][0][2])
            correct = np.sum((hs-ps)**2) + np.sum((ht-pt)**2)
            swapped = np.sum((hs-pt)**2) + np.sum((ht-ps)**2)
            if swapped < correct-1e-6:
                return True
    return False


def _recover(input_grid):
    """Return the unique pre-shear board and input-derived facts.

    Search all rectangular-border candidates, including boards with structural
    islands. Equal congruence classes must have one color, so all exact covers
    yield the same colored board. Ambiguity is rejected before shear, even if
    clipping could conceal it.
    """
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= a.size//2:
        raise ValueError("background must occupy a strict majority of the input")
    solutions = []
    for structure in range(10):
        if structure == background:
            continue
        for component in _components(a == structure):
            r0, r1, c0, c1 = bounds = _bbox(component)
            blueprint = a[r0:r1, c0:c1]
            if min(blueprint.shape) < 5:
                continue
            if not (np.all(blueprint[0] == structure)
                    and np.all(blueprint[-1] == structure)
                    and np.all(blueprint[:, 0] == structure)
                    and np.all(blueprint[:, -1] == structure)
                    and np.all((blueprint == structure) | (blueprint == background))):
                continue
            holes = _components(blueprint == background)
            if len(holes) < 2:
                continue
            outside = np.ones(a.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            moat = np.zeros(a.shape, dtype=bool)
            moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
            if np.any(moat & outside & (a != background)):
                continue
            external = [(int(a[cells[0]]), cells)
                        for cells in _components(outside & (a != background))]
            if any(len({int(a[r, c]) for r, c in cells}) != 1
                   or color == structure for color, cells in external):
                continue
            pieces = [(color, cells) for color, cells in external if len(cells) >= 4]
            noise = [(color, cells) for color, cells in external if len(cells) < 4]
            if len(pieces) != len(holes):
                continue
            hole_area = sum(map(len, holes))
            piece_area = sum(len(cells) for color, cells in pieces)
            # Area ratio plus shape checks avoid a guessed or memorized scale.
            scales = [k for k in range(2, 31//max(blueprint.shape)+1)
                      if hole_area*k*k == piece_area]
            hg = defaultdict(list)
            for hole in holes:
                hg[_canonical(hole)].append(hole)
            for k in scales:
                decoded = [(color, _decode(cells, k), cells) for color, cells in pieces]
                if any(shape is None for color, shape, cells in decoded):
                    continue
                pg = defaultdict(list)
                for color, shape, cells in decoded:
                    pg[_canonical(shape)].append((color, shape, cells))
                if set(pg) != set(hg) or any(len(pg[s]) != len(hg[s]) for s in hg):
                    continue
                if any(len({p[0] for p in group}) != 1 for group in pg.values()):
                    raise ValueError("interchangeable pieces give different colored assemblies")
                logical = blueprint.copy()
                for shape, hs in hg.items():
                    for hole in hs:
                        for r, c in hole:
                            logical[r, c] = pg[shape][0][0]
                board = _expand(logical, k)
                blank = _expand(blueprint, k)
                answer = _shear(board, background)
                rotation = any(len(hg[s]) == 1 and _normal(hg[s][0]) != pg[s][0][1]
                               for s in hg)
                classes = tuple(sorted(hg))
                equal_area = any(len(s) == len(t) and s != t
                                 and pg[s][0][0] != pg[t][0][0]
                                 and np.any(answer == pg[s][0][0])
                                 and np.any(answer == pg[t][0][0])
                                 for i, s in enumerate(classes) for t in classes[i+1:])
                info = dict(background=background, structure=structure, scale=k,
                            bounds=bounds, blueprint=blueprint.copy(), logical=logical,
                            holes=holes, pieces=decoded, classes=classes,
                            rotation=rotation, equal_area=equal_area,
                            proximity_conflict=_proximity_conflict(hg, pg, bounds, answer),
                            visible_pixels=tuple(_visible_pixels(h, k, board.shape[0]) for h in holes),
                            noise_sizes=tuple(len(cells) for color, cells in noise),
                            repeated=any(len(g) > 1 for g in hg.values()),
                            board_pattern=_key(blueprint == structure),
                            blank_sheared=_shear(blank, background))
                solutions.append((board, info))
    if len(solutions) != 1:
        raise ValueError("expected exactly one blueprint, common scale and colored assembly")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    board, info = _recover(input_grid)
    return _shear(board, info["background"])


def _grow(rng, area, mh, mw):
    cells = {(0, 0)}
    style = int(rng.integers(3))
    while len(cells) < area:
        choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in NEIGHBORS} - cells)
        choices = [p for p in choices if all(x <= y for x, y in
                   zip(_size(_normal(cells | {p})), (mh, mw)))]
        if not choices:
            return None
        weights = []
        for r, c in choices:
            neighbors = sum((r+dr, c+dc) in cells for dr, dc in NEIGHBORS)
            h, w = _size(_normal(cells | {(r, c)}))
            weights.append((1 + max(h, w)/min(h, w)) if style == 0
                           else 3**neighbors if style == 1 else 1/neighbors)
        p = np.asarray(weights, dtype=float)
        cells.add(choices[int(rng.choice(len(choices), p=p/p.sum()))])
    return _normal(cells)


def _sample_shapes(rng, count, bh, bw, k, witness):
    limit = min(5, 15//k, bh-2, bw-2)
    max_area = min(9, limit*limit, 7 if k >= 3 else 9)
    shapes = []
    equal_area = int(rng.integers(3, min(6, max_area)+1))
    for i in range(count):
        for _ in range(80):
            if not witness and shapes and rng.random() < .2:
                candidate = shapes[int(rng.integers(len(shapes)))]
            else:
                area = equal_area if witness and i < 2 else int(rng.integers(1, max_area+1))
                candidate = _grow(rng, area, limit, limit)
            if candidate is None:
                continue
            if witness and i == 0 and len(_rotations(candidate)) == 1:
                continue
            if witness and i == 1 and _canonical(candidate) == _canonical(shapes[0]):
                continue
            if witness and i >= 2 and _canonical(candidate) in {_canonical(s) for s in shapes[:2]}:
                continue
            shapes.append(candidate)
            break
        else:
            return None
    return shapes


def _place_holes(rng, shapes, bh, bw):
    for _ in range(8):
        used, placed = set(), [None]*len(shapes)
        for index in sorted(range(len(shapes)), key=lambda i: (-len(shapes[i]), i)):
            shape = shapes[index]
            sh, sw = _size(shape)
            origins = [(r, c) for r in range(1, bh-sh) for c in range(1, bw-sw)]
            for j in rng.permutation(len(origins)):
                rr, cc = origins[int(j)]
                cells = tuple((r+rr, c+cc) for r, c in shape)
                if any(p in used or any((p[0]+dr, p[1]+dc) in used for dr, dc in NEIGHBORS)
                       for p in cells):
                    continue
                placed[index] = cells
                used.update(cells)
                break
            if placed[index] is None:
                break
        else:
            return placed
    return None


def _origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return np.argwhere(sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w] == 0)


def _pack(rng, blueprint, loose, colors, k, bg, noise_color):
    masks = [np.ones(blueprint.shape, dtype=bool)] + [_expand(_mask(s), k) for s in loose]
    for attempt in range(16):
        height, width = ((int(rng.integers(18, 31)), int(rng.integers(18, 31)))
                         if attempt < 10 else (30, 30))
        canvas = np.full((height, width), bg, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = [0] + sorted(range(1, len(masks)), key=lambda i: -masks[i].size)
        for i in order:
            h, w = masks[i].shape
            origins = _origins(blocked, h, w)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            view = canvas[r:r+h, c:c+w]
            if i == 0:
                view[:] = blueprint
            else:
                view[masks[i]] = colors[i-1]
            blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
        else:
            target = 0 if rng.random() < .3 else int(rng.integers(1, 13))
            for _ in range(target*2+1):
                if target <= 0:
                    break
                size = min(target, int(rng.integers(1, 4)))
                shape = ((0, 0),) if size == 1 else (((0, 0), (0, 1)) if size == 2
                          else ((0, 0), (1, 0), (1, 1)))
                variants = _rotations(shape)
                shape = variants[int(rng.integers(len(variants)))]
                h, w = _size(shape)
                positions = _origins(blocked, h, w)
                if not len(positions):
                    continue
                r, c = map(int, positions[int(rng.integers(len(positions)))])
                color = int(rng.choice(colors)) if rng.random() < .25 else noise_color
                for dr, dc in shape:
                    canvas[r+dr, c+dc] = color
                blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
                target -= size
            if target > 0 or np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            # Crop only if the visible background stays a strict majority.
            if rng.random() < .65:
                r0, r1, c0, c1 = _bbox(np.argwhere(canvas != bg))
                r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
                r1, c1 = min(height, r1+int(rng.integers(4))), min(width, c1+int(rng.integers(4)))
                crop = canvas[r0:r1, c0:c1]
                if np.count_nonzero(crop == bg) > crop.size//2:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, k, witness=False, novel_against=frozenset()):
    for _ in range(180):
        palette = list(map(int, rng.permutation(10)))
        bg, structure, noise_color = palette[:3]
        maximum = min(12, 30//k)
        bh, bw = map(int, rng.integers(5, maximum+1, size=2))
        count = int(rng.integers(2, min(6, max(2, (bh-2)*(bw-2)//5))+1))
        shapes = _sample_shapes(rng, count, bh, bw, k, witness)
        if shapes is None:
            continue
        if novel_against and not any(_canonical(s) not in novel_against for s in shapes):
            continue
        placements = _place_holes(rng, shapes, bh, bw)
        if placements is None:
            continue
        classes = list(dict.fromkeys(_canonical(s) for s in shapes))
        color_count = len(classes) if witness else int(rng.integers(1, len(classes)+1))
        colors_by_class = {s: palette[3+i % color_count] for i, s in enumerate(classes)}
        colors = [colors_by_class[_canonical(s)] for s in shapes]
        blueprint = np.full((bh, bw), structure, dtype=np.int8)
        logical = blueprint.copy()
        loose = []
        for i, (cells, color) in enumerate(zip(placements, colors)):
            destination = _normal(cells)
            variants = _rotations(destination)
            if witness and i == 0:
                variants = tuple(s for s in variants if s != destination)
            if not variants:
                break
            loose.append(variants[int(rng.integers(len(variants)))])
            for r, c in cells:
                blueprint[r, c] = bg
                logical[r, c] = color
        else:
            board = _expand(logical, k)
            output = _shear(board, bg)
            # At least one piece survives; in a witness the two equal-area
            # pieces of different shape/color must both survive clipping.
            required_colors = colors[:2] if witness else colors[:1]
            if any(not np.any(output == color) for color in required_colors):
                continue
            if novel_against and not any(
                    _canonical(hole) not in novel_against
                    and 2*_visible_pixels(hole, k, board.shape[0]) >= len(hole)*k*k
                    for hole in placements):
                continue
            canvas = _pack(rng, blueprint, loose, colors, k, bg, noise_color)
            if canvas is None:
                continue
            recovered, info = _recover(canvas)
            if not np.array_equal(recovered, board):
                raise AssertionError("construction and input recovery disagree")
            if witness and not (info["rotation"] and info["equal_area"] and info["proximity_conflict"]):
                continue
            return {"input": canvas, "output": output}
    raise RuntimeError("bounded blueprint construction exhausted")


def _count(n, low, high, name):
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or not low <= n <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(n)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 2, 10, "num_train")
    ntest = _count(num_test, 1, 4, "num_test")
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(4))
    if mode < 2:
        a, b = (2, 3) if mode == 0 else (3, 2)
        train_scales, test_scales = [a]*ntrain, [b]*ntest
    else:
        train_scales = [2, 3] + [int(rng.choice([2, 3])) for _ in range(ntrain-2)]
        rng.shuffle(train_scales)
        test_scales = [4]*ntest if mode == 2 else [int(rng.choice([2, 3])) for _ in range(ntest)]
    puzzle, seen, training_classes = {"train": [], "test": []}, set(), set()
    training_patterns = set()
    for split, scales in (("train", train_scales), ("test", test_scales)):
        for i, k in enumerate(scales):
            for _ in range(12):
                pair = _make_pair(rng, k, witness=split == "train" and i == 0,
                                  novel_against=training_classes if split == "test" else frozenset())
                key = _key(pair["input"])
                info = _recover(pair["input"])[1]
                if key not in seen and not (split == "train" and i == 1
                                           and info["board_pattern"] in training_patterns):
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_classes.update(info["classes"])
                training_patterns.add(info["board_pattern"])
    # Training order does not identify the mandatory witness.
    rng.shuffle(puzzle["train"])
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    board, info = _recover(a)
    bg, k = info["background"], info["scale"]
    assert np.array_equal(out, _shear(board, bg)), "incorrect sheared assembly"
    assert np.array_equal(out[-1], board[-1]), "bottom row must be fixed"
    assert not np.array_equal(out, board), "omitting A must change the output"
    assert not np.array_equal(out, info["blank_sheared"]), "omitting B must change the output"
    assert not np.any(board == bg), "pre-shear board must be completely filled"
    h, w = board.shape
    expected_blank = sum(min(w, h-1-r) for r in range(h))
    assert np.count_nonzero(out == bg) == expected_blank, "clipping triangle/trapezoid"
    for r in range(h):
        d = h-1-r
        if d < w:
            assert np.array_equal(out[r, :w-d], board[r, d:])
            assert np.all(out[r, w-d:] == bg)
        else:
            assert np.all(out[r] == bg)
    assert 2 <= k <= 4 and 2 <= len(info["holes"]) <= 6
    assert all(n <= 3 for n in info["noise_sizes"])
    assert all(1 <= len(s) <= 9 for s in info["classes"])
    assert sum(len(cells) for color, shape, cells in info["pieces"]) == sum(map(len, info["holes"]))*k*k
    # Reversing the operations would preserve k-row blocks. Correct shear must
    # split the k-row structural band into unequal physical rows at the bottom.
    assert not np.array_equal(out[-1], out[-2])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count(len(puzzle["train"]), 2, 10, "num_train")
    _count(len(puzzle["test"]), 1, 4, "num_test")
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            facts[split].append(_check_pair(pair))
    assert any(i["rotation"] and i["equal_area"] and i["proximity_conflict"] for i in facts["train"])
    training_classes = set(s for i in facts["train"] for s in i["classes"])
    assert all(any(_canonical(h) not in training_classes
                   and 2*v >= len(h)*i["scale"]**2
                   for h, v in zip(i["holes"], i["visible_pixels"])) for i in facts["test"])
    assert len({i["board_pattern"] for i in facts["train"]}) >= 2


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(p[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _negative_tests():
    # Reflection is not a valid substitute for rotation. One L and its mirror
    # share area and D4 shape but are different rotation-only classes.
    ell = ((0, 0), (1, 0), (2, 0), (2, 1))
    mirror = _normal([(r, -c) for r, c in ell])
    assert _canonical(ell) != _canonical(mirror)
    reflected = np.zeros((24, 24), dtype=np.int8)
    reflected[:6, :7] = 2
    for r, c in ell:
        reflected[r+1, c+1] = 0
    reflected[1, 5] = 0
    reflected[10:16, :4][_expand(_mask(mirror), 2)] = 3
    reflected[10:12, 8:10] = 4
    try:
        solve(reflected)
    except ValueError:
        pass
    else:
        raise AssertionError("reflection-only match accepted")
    reflected[10:16, :4] = 0
    reflected[10:16, :4][_expand(_mask(ell), 2)] = 3
    assert solve(reflected).shape == (12, 14)
    # A completely explicit row example, including top rows clipped to nothing.
    board = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 2], [3, 4, 5]])
    assert np.array_equal(_shear(board, 0), [[0, 0, 0], [7, 0, 0], [9, 2, 0], [3, 4, 5]])
    # Uniqueness and partial macrocell corruption, checked by input recovery.
    a = np.zeros((24, 24), dtype=np.int8)
    a[:7, :8] = 2
    for r, c in ((1, 1), (1, 5)):
        a[r, c] = 0
    a[10:12, :2] = 3
    a[10:12, 6:8] = 4
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colors accepted")
    a[a == 4] = 3
    assert solve(a).shape == (14, 16)
    a[10, 0] = 0
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError("broken block accepted")


def self_test():
    """42 fixed seeds, one puzzle in memory at a time; no files are written."""
    metrics = {name: set() for name in ("inputs", "outputs", "boards", "classes", "scales",
                                      "counts", "backgrounds", "structures", "aspects", "directions")}
    digests, areas, noise_sizes = set(), set(), set()
    repeated = clipped_whole_rows = rotation_pairs = equal_pairs = proximity_pairs = 0
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = _puzzle_digest(p)
        assert digest == _puzzle_digest(generate_puzzle(seed)), "nondeterministic generation"
        digests.add(digest)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _check_pair(pair)
                inp, out = pair["input"], pair["output"]
                metrics["inputs"].add(inp.shape)
                metrics["outputs"].add(out.shape)
                metrics["boards"].add(info["board_pattern"])
                metrics["classes"].update(info["classes"])
                metrics["scales"].add(info["scale"])
                metrics["counts"].add(len(info["holes"]))
                metrics["backgrounds"].add(info["background"])
                metrics["structures"].add(info["structure"])
                h, w = out.shape
                metrics["aspects"].add("square" if h == w else "tall" if h > w else "wide")
                r0, r1, c0, c1 = info["bounds"]
                for color, shape, cells in info["pieces"]:
                    top, bottom, left, right = _bbox(cells)
                    for condition, name in ((top < r0, "above"), (bottom > r1, "below"),
                                            (left < c0, "left"), (right > c1, "right")):
                        if condition:
                            metrics["directions"].add(name)
                areas.update(map(len, info["classes"]))
                noise_sizes.update(info["noise_sizes"])
                repeated += info["repeated"]
                rotation_pairs += info["rotation"]
                equal_pairs += info["equal_area"]
                proximity_pairs += info["proximity_conflict"]
                clipped_whole_rows += h > w
                assert np.array_equal(solve(inp.copy()), out)
                pairs += 1
    assert len(digests) == 42
    assert metrics["scales"] == {2, 3, 4}
    assert metrics["counts"] >= {2, 3, 4}
    assert len(metrics["classes"]) >= 35 and len(metrics["boards"]) >= 150
    assert len(metrics["inputs"]) >= 35 and len(metrics["outputs"]) >= 25
    assert len(metrics["backgrounds"]) >= 8 and len(metrics["structures"]) >= 8
    assert metrics["aspects"] == {"square", "wide", "tall"}
    assert metrics["directions"] == {"above", "below", "left", "right"}
    assert {1, 2, 3, 4, 5, 6} <= areas and noise_sizes == {1, 2, 3}
    assert repeated > 0 and clipped_whole_rows > 0 and rotation_pairs >= 42 and equal_pairs >= 42
    assert proximity_pairs >= 42
    _negative_tests()
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            q = generate_puzzle(1000+10*train+test, train, test)
            validate_generated_puzzle(q)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs,
            "distinct_puzzles": len(digests),
            "variation": {name: len(values) for name, values in metrics.items()},
            "shape_areas": sorted(areas), "noise_sizes": sorted(noise_sizes),
            "repeated_class_pairs": repeated, "whole_row_clipping_pairs": clipped_whole_rows,
            "rotation_pairs": rotation_pairs, "equal_area_witness_pairs": equal_pairs,
            "proximity_counterexample_pairs": proximity_pairs}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
