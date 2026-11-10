"""Material ballots around cracks. Standard library + NumPy, no external state.

The clean scene is recovered from large surviving material components and a
unique fitting overlap order. Crack ballots are computed only after recovery.
Construction and prediction use separate rim composition implementations.
"""
from itertools import permutations
import numpy as np


ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, points = [(r, c)], []
        seen[r, c] = True
        while stack:
            p = stack.pop()
            points.append(p)
            for dr, dc in ORTHOGONAL:
                q = p[0] + dr, p[1] + dc
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and mask[q] and not seen[q]):
                    seen[q] = True
                    stack.append(q)
        result.append(points)
    return result


def _mask(shape, points):
    a = np.zeros(shape, dtype=bool)
    if points:
        r, c = zip(*points)
        a[r, c] = True
    return a


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _box(points, color):
    p = np.asarray(points)
    r, c = p.min(axis=0)
    rr, cc = p.max(axis=0) + 1
    return int(r), int(c), int(rr), int(cc), int(color)


def _paint(shape, boxes, order):
    clean = np.zeros(shape, dtype=np.int64)
    owner = np.full(shape, -1, dtype=np.int64)
    for i in order:
        r, c, rr, cc, color = boxes[i]
        clean[r:rr, c:cc] = color
        owner[r:rr, c:cc] = i
    return clean, owner


def _recover_scene(inp):
    """Enumerate every order of the input-supported bounding rectangles.

    All large components must fit, small components must be exterior, and
    erasures must leave connected visible pieces and intact extent evidence.
    Distinct admissible visible scenes are rejected rather than tie-broken.
    """
    fills = tuple(int(c) for c in np.unique(inp) if c != 0)
    if len(fills) != 2 or 0 not in inp:
        raise ValueError("The family uses background 0 and two nonzero fills")
    cores, boxes, specks = [], [], []
    for color in fills:
        for points in _components(inp == color):
            if len(points) <= 3:
                specks.extend(points)
                continue
            if len(points) < 16:
                raise ValueError("Unsupported intermediate-size material component")
            box = _box(points, color)
            r, c, rr, cc, _ = box
            if min(rr-r, cc-c) < 6:
                raise ValueError("Material rectangles need six-cell extents")
            a = np.asarray(points)
            if min(np.count_nonzero(a[:, 0] == r),
                   np.count_nonzero(a[:, 0] == rr-1),
                   np.count_nonzero(a[:, 1] == c),
                   np.count_nonzero(a[:, 1] == cc-1)) < 2:
                raise ValueError("Each rectangle extent needs two surviving cells")
            cores.append(points)
            boxes.append(box)
    if not 2 <= len(boxes) <= 4:
        raise ValueError("Expected two to four supported rectangles")
    if {b[4] for b in boxes} != set(fills):
        raise ValueError("Both materials need large surviving components")
    union = np.zeros(inp.shape, dtype=bool)
    for r, c, rr, cc, _ in boxes:
        union[r:rr, c:cc] = True
    if len(_components(union)) != 1:
        raise ValueError("Materials must form one four-connected shared scene")
    core_indices = [(np.array([p[0] for p in pts]),
                     np.array([p[1] for p in pts])) for pts in cores]
    small = _mask(inp.shape, specks)
    scenes = {}
    for order in permutations(range(len(boxes))):
        clean, owner = _paint(inp.shape, boxes, order)
        if np.any(clean[small] != 0):
            continue
        valid = True
        for i, indices in enumerate(core_indices):
            if not np.all(owner[indices] == i):
                valid = False
                break
            visible = owner == i
            area = int(visible.sum())
            holes = visible & (inp == 0)
            if (area < 20 or int(holes.sum()) > area // 5
                    or len(_components(visible)) != 1):
                valid = False
                break
            # No wrong-material errors, attached noise, or merged identities.
            observed = visible & (inp != 0)
            if int(observed.sum()) != len(cores[i]):
                valid = False
                break
        if valid:
            scenes[clean.tobytes()] = clean
    if len(scenes) != 1:
        raise ValueError("No unique supported visible rectangle composition")
    return next(iter(scenes.values())), fills, tuple(boxes)


def _ballots(clean, holes, fills):
    result = []
    for points in _components(holes):
        values = [int(clean[p]) for p in points]
        counts = tuple(values.count(c) for c in fills)
        winner = 0 if counts[0] == counts[1] else fills[int(counts[1] > counts[0])]
        rim_color = 0 if winner == 0 else fills[1] if winner == fills[0] else fills[0]
        region = _mask(clean.shape, points)
        rim = _dilate(region) & ~holes
        result.append((points, counts, rim_color, rim))
    return result


def _prediction(clean, holes, fills):
    # Bits represent color demands, independent of component traversal order.
    demands = np.zeros(clean.shape, dtype=np.uint8)
    for _, _, color, rim in _ballots(clean, holes, fills):
        bit = 4 if color == 0 else 1 if color == fills[0] else 2
        demands[rim] |= bit
    out = clean.copy()
    out[demands == 1] = fills[0]
    out[demands == 2] = fills[1]
    out[(demands != 0) & (demands != 1) & (demands != 2)] = 0
    out[holes] = 0
    return out, demands


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction; roles and ballots have no puzzle/seed state."""
    inp = _grid(input_grid)
    clean, fills, _ = _recover_scene(inp)
    holes = (clean != 0) & (inp == 0)
    return _prediction(clean, holes, fills)[0]


def _constructed_output(clean, holes):
    """Construction renderer uses explicit geometry and set-valued demands."""
    fills = tuple(int(c) for c in np.unique(clean) if c != 0)
    demands = {}
    h, w = clean.shape
    all_holes = set(map(tuple, np.argwhere(holes)))
    for points in _components(holes):
        a = sum(int(clean[p]) == fills[0] for p in points)
        b = len(points) - a
        color = 0 if a == b else fills[1] if a > b else fills[0]
        for r, c in points:
            for rr in range(max(0, r-1), min(h, r+2)):
                for cc in range(max(0, c-1), min(w, c+2)):
                    q = rr, cc
                    if q not in all_holes:
                        demands.setdefault(q, set()).add(color)
    out = clean.copy()
    for p, colors in demands.items():
        out[p] = next(iter(colors)) if len(colors) == 1 else 0
    out[holes] = 0
    return out


def _local_output(clean, holes, fills):
    """A deliberate shortcut: choose the complement separately at each hole."""
    demands = np.zeros(clean.shape, dtype=np.uint8)
    for color, bit in ((fills[0], 2), (fills[1], 1)):
        demands[_dilate(holes & (clean == color)) & ~holes] |= bit
    out = clean.copy()
    out[demands == 1] = fills[0]
    out[demands == 2] = fills[1]
    out[demands == 3] = 0
    out[holes] = 0
    return out


def _features(inp):
    clean, fills, boxes = _recover_scene(inp)
    holes = (clean != 0) & (inp == 0)
    ballots = _ballots(clean, holes, fills)
    out, demands = _prediction(clean, holes, fills)
    pure = {fills[int(counts[1] > 0)] for _, counts, _, _ in ballots if 0 in counts}
    mixed = [(points, counts, rim) for points, counts, _, rim in ballots if min(counts) > 0]
    edge = np.zeros(inp.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    diagonal = bool((holes[1:, 1:] & holes[:-1, :-1]).any()
                    or (holes[1:, :-1] & holes[:-1, 1:]).any())
    branching = any(sum((p[0]+dr, p[1]+dc) in set(points)
                        for dr, dc in ORTHOGONAL) >= 3
                    for points, _, _, _ in ballots for p in points)
    border_disagrees = False
    for _, counts, rim in mixed:
        if counts[0] != counts[1]:
            border_counts = tuple(int(np.count_nonzero(clean[rim] == c)) for c in fills)
            border_disagrees |= int(np.sign(counts[0]-counts[1])) != int(np.sign(border_counts[0]-border_counts[1]))
    return {
        "clean": clean, "fills": fills, "boxes": boxes, "holes_mask": holes,
        "expected": out, "ballots": ballots, "holes": int(holes.sum()),
        "groups": len(ballots), "largest": max((len(p) for p, _, _, _ in ballots), default=0),
        "pure_both": pure == set(fills),
        "mixed_majority": any(a != b for _, (a, b), _ in mixed),
        "tie": any(a == b for _, (a, b), _ in mixed),
        "opposing_rims": bool(np.any(demands == 3)),
        "tie_color_contact": bool(np.any(((demands & 4) != 0)
                                         & ((demands & 3) != 0) & (clean != 0))),
        "diagonal": diagonal, "branching": branching,
        "adjacent": any(len(p) > 1 for p, _, _, _ in ballots),
        "clipped": bool((holes & edge).any()),
        "extras": int(np.count_nonzero((clean == 0) & (inp != 0))),
        "local_shortcut_differs": not np.array_equal(out, _local_output(clean, holes, fills)),
        "border_shortcut_differs": bool(border_disagrees),
    }


def _allocate(total, count, minimum, rng):
    values = [minimum] * count
    for _ in range(total - count * minimum):
        values[int(rng.integers(count))] += 1
    return values


def _sample_clean(rng, h, w, count, mode):
    """Independent geometry samplers, all gated by visible input evidence."""
    boxes = []
    if mode == 0:  # independently sized, offset overlapping rectangles
        for i in range(count):
            for _ in range(100):
                rh = int(rng.integers(8, h+1))
                rw = int(rng.integers(8, w+1))
                r = int(rng.integers(h-rh+1))
                c = int(rng.integers(w-rw+1))
                b = r, c, r+rh, c+rw, 1+i % 2
                if boxes:
                    def overlap(x):
                        return max(0, min(b[2], x[2])-max(b[0], x[0])) * max(0, min(b[3], x[3])-max(b[1], x[1]))
                    if overlap(boxes[-1]) < 4 or any(overlap(x) for x in boxes[:-1]):
                        continue
                boxes.append(b)
                break
            else:
                return None
    elif mode == 1:  # nested material layers, including a recurring inner fill
        if count > 3 or min(h, w) < (26 if count == 3 else 18):
            return None
        r, c = int(rng.integers(3)), int(rng.integers(3))
        rr, cc = h-int(rng.integers(3)), w-int(rng.integers(3))
        boxes.append((r, c, rr, cc, 1))
        for i in range(1, count):
            margins = [int(rng.integers(3, 5)) for _ in range(4)]
            r, c, rr, cc = r+margins[0], c+margins[1], rr-margins[2], cc-margins[3]
            if min(rr-r, cc-c) < 6:
                return None
            boxes.append((r, c, rr, cc, 1+i % 2))
    else:  # freely sized staircase of overlapping slabs, not a fixed zigzag
        lo, hi = int(rng.integers(3)), w-int(rng.integers(3))
        if hi-lo < count*5:
            return None
        widths = _allocate(hi-lo, count, 5, rng)
        cuts = [lo]
        for size in widths:
            cuts.append(cuts[-1]+size)
        for i in range(count):
            rh = int(rng.integers(8, h+1))
            r = int(rng.integers(h-rh+1))
            c = max(lo, cuts[i]-int(rng.integers(1, 3)))
            cc = min(hi, cuts[i+1]+int(rng.integers(1, 3)))
            boxes.append((r, c, r+rh, cc, 1+i % 2))
        # Require an overlapping shared scene rather than separate puzzles.
        for a, b in zip(boxes, boxes[1:]):
            if min(a[2], b[2])-max(a[0], b[0]) < 3:
                return None
    order = list(range(count)) if mode == 1 else list(map(int, rng.permutation(count)))
    clean, _ = _paint((h, w), boxes, order)
    if len(_components(clean != 0)) != 1 or not np.any(clean == 0):
        return None
    try:
        recovered, _, inferred = _recover_scene(clean)
    except ValueError:
        return None
    if len(inferred) != count or not np.array_equal(recovered, clean):
        return None
    return clean


def _seams(clean, diagonal=False, distance=1):
    offsets = ((distance, distance), (distance, -distance)) if diagonal else ((0, distance), (distance, 0))
    h, w = clean.shape
    pairs = []
    for r, c in np.argwhere(clean != 0):
        p = int(r), int(c)
        for dr, dc in offsets:
            q = p[0]+dr, p[1]+dc
            if 0 <= q[0] < h and 0 <= q[1] < w and clean[q] and clean[p] != clean[q]:
                pairs.append((p, q))
    return pairs


def _can_add(clean, holes, points, allow_edge=True):
    h, w = clean.shape
    forbidden = _dilate(_dilate(holes))
    return all(0 <= r < h and 0 <= c < w and clean[r, c] != 0
               and not forbidden[r, c]
               and (allow_edge or (0 < r < h-1 and 0 < c < w-1))
               for r, c in points)


def _place_group(clean, holes, proposals, rng, allow_edge=True):
    for i in rng.permutation(len(proposals)):
        points = proposals[int(i)]
        if _can_add(clean, holes, points, allow_edge):
            r, c = zip(*points)
            holes[r, c] = True
            return True
    return False


def _majority_proposals(clean):
    proposals = []
    for p, q in _seams(clean):
        for a, b in ((p, q), (q, p)):
            for dr, dc in ORTHOGONAL:
                z = a[0]+dr, a[1]+dc
                if (0 <= z[0] < clean.shape[0] and 0 <= z[1] < clean.shape[1]
                        and z != b and clean[z] == clean[a]):
                    proposals.append([a, b, z])
    return proposals


def _tie_contact_proposals(clean):
    """A tied seam dimer and a separate pure ballot with a shared rim.

    The diagonal singleton is never orthogonally joined to either dimer cell.
    Thus its material demand meets a background demand without changing votes.
    """
    proposals = []
    h, w = clean.shape
    for p, q in _seams(clean):
        for r, c in (p, q):
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                z = r+dr, c+dc
                if (0 <= z[0] < h and 0 <= z[1] < w and clean[z] != 0
                        and all(abs(z[0]-a[0])+abs(z[1]-a[1]) > 1
                                for a in (p, q))):
                    proposals.append([p, q, z])
    return proposals


def _grow_crack(clean, holes, rng, length, mixed, allow_edge):
    if mixed:
        starts = [[p, q] for p, q in _seams(clean)]
    else:
        starts = [[tuple(map(int, p))] for p in np.argwhere(clean != 0)]
    for i in rng.permutation(len(starts))[:80]:
        points = list(starts[int(i)])
        if not _can_add(clean, holes, points, allow_edge):
            continue
        while len(points) < length:
            choices = sorted({(r+dr, c+dc) for r, c in points for dr, dc in ORTHOGONAL}
                             - set(points))
            choices = [p for p in choices if _can_add(clean, holes, [p], allow_edge)
                       and (mixed or clean[p] == clean[points[0]])]
            if not choices:
                break
            points.append(choices[int(rng.integers(len(choices)))])
        if len(points) == length:
            r, c = zip(*points)
            holes[r, c] = True
            return True
    return False


def _make_holes(clean, rng, style, allow_edge):
    holes = np.zeros(clean.shape, dtype=bool)
    if style == "directions":
        # Diagonal contact creates two ballots, not one eight-connected crack.
        if not _place_group(clean, holes, [[p, q] for p, q in _seams(clean, True)], rng):
            return None
        dimers = []
        for r, c in np.argwhere(clean != 0):
            p = int(r), int(c)
            for dr, dc in ((0, 1), (1, 0)):
                q = p[0]+dr, p[1]+dc
                if q[0] < clean.shape[0] and q[1] < clean.shape[1] and clean[q] == clean[p]:
                    dimers.append([p, q])
        if not _place_group(clean, holes, dimers, rng):
            return None
        edge = [[tuple(map(int, p))] for p in np.argwhere(clean != 0)
                if p[0] in (0, clean.shape[0]-1) or p[1] in (0, clean.shape[1]-1)]
        if allow_edge and edge:
            _place_group(clean, holes, edge, rng)
    elif style == "ballots":
        if not _place_group(clean, holes, _majority_proposals(clean), rng, allow_edge):
            return None
        if not _place_group(clean, holes, _tie_contact_proposals(clean), rng, allow_edge):
            return None
    elif style == "transfer":
        # Odd length guarantees a strict ballot; a seam guarantees both fills.
        if not _grow_crack(clean, holes, rng, int(rng.choice((7, 9))), True, allow_edge):
            return None
        for _ in range(int(rng.integers(1, 4))):
            _grow_crack(clean, holes, rng, int(rng.integers(1, 5)), bool(rng.integers(2)), allow_edge)
    else:
        for _ in range(int(rng.integers(1, 6))):
            length = int(rng.integers(1, 6))
            _grow_crack(clean, holes, rng, length, length >= 2 and rng.random() < .55, allow_edge)
    if style in ("transfer", "ordinary") and rng.random() < .55:
        # Distinct ballots may meet diagonally or share rims at distance two.
        contacts = _seams(clean, bool(rng.integers(2)), int(rng.integers(1, 3)))
        _place_group(clean, holes, [[p, q] for p, q in contacts], rng, allow_edge)
    return holes if holes.any() else None


def _add_specks(inp, clean, rng, required):
    available = ~_dilate(clean != 0)
    points = [tuple(map(int, p)) for p in np.argwhere(available)]
    if required and not points:
        return None
    count = int(rng.integers(1 if required else 0, min(8, len(points))+1))
    rng.shuffle(points)
    fills = [int(c) for c in np.unique(clean) if c]
    for p in points[:count]:
        inp[p] = fills[int(rng.integers(2))]
    return inp


def _make_pair(rng, style, palette, allow_edge=True):
    for _ in range(160):
        count = int(rng.choice((2, 3, 4), p=(.58, .30, .12)))
        if style in ("directions", "ballots"):
            count = 2
        low = 22 if style == "transfer" else 18 if style in ("directions", "ballots") else 14
        h, w = int(rng.integers(low, 31)), int(rng.integers(max(low, count*5+2), 31))
        mode = int(rng.choice((0, 1, 2), p=(.45, .20, .35)))
        clean = _sample_clean(rng, h, w, count, mode)
        if clean is None:
            continue
        for _ in range(16):
            holes = _make_holes(clean, rng, style, allow_edge)
            if holes is None:
                continue
            inp = clean.copy()
            inp[holes] = 0
            inp = _add_specks(inp, clean, rng, style == "directions")
            if inp is None:
                continue
            try:
                f = _features(inp)
            except ValueError:
                continue
            if not np.array_equal(f["clean"], clean):
                continue
            if style == "directions" and not (f["pure_both"] and f["diagonal"] and f["opposing_rims"] and f["adjacent"]):
                continue
            if style == "ballots" and not (f["mixed_majority"] and f["tie"] and f["tie_color_contact"] and f["local_shortcut_differs"] and f["border_shortcut_differs"]):
                continue
            if style == "transfer" and not (f["largest"] >= 7 and f["mixed_majority"] and f["local_shortcut_differs"]):
                continue
            mapping = np.array([0, *palette])
            actual_clean, actual_input = mapping[clean], mapping[inp]
            out = _constructed_output(actual_clean, holes)
            turns = int(rng.integers(4))
            reflected = bool(rng.integers(2))
            def transform(a):
                return np.rot90(np.fliplr(a) if reflected else a, turns).copy()
            pair = {"input": transform(actual_input), "output": transform(out)}
            if not np.array_equal(solve(pair["input"]), pair["output"]):
                raise AssertionError("Construction and input-only prediction disagree")
            return pair
    raise RuntimeError("Unable to construct readable crack-ballot evidence")


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("Illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
    varying_palette = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            style = ("directions" if i == 0 else "ballots" if i == 1 else "ordinary") if split == "train" else "transfer"
            palette = tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False))) if varying_palette else stable_palette
            # Clipping is allowed in tests only if a demonstration exposes it.
            allow_edge = split == "train" or any(_features(p["input"])["clipped"] for p in puzzle["train"])
            for _ in range(20):
                pair = _make_pair(rng, style, palette, allow_edge)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unable to produce a distinct input")
    # Shuffle anchors so demonstration index itself has no semantic meaning.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("Puzzle must contain exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise AssertionError("Splits must be lists")
    try:
        _check_counts(len(puzzle["train"]), len(puzzle["test"]))
        seen, features = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                assert inp.shape == out.shape
                key = _signature(inp)
                assert key not in seen, "Duplicate input"
                seen.add(key)
                f = _features(inp)
                assert f["holes"] > 0, "Every example must contain a crack"
                assert len(_components(f["clean"] != 0)) == 1, "Materials must form one shared scene"
                assert np.array_equal(f["expected"], out), "Input-inferred label mismatch"
                assert not np.array_equal(inp, out)
                # A's negative shape stays open even where other rims reach it.
                assert np.all(out[f["holes_mask"]] == 0)
                assert set(map(int, np.unique(out))) <= {0, *f["fills"]}
                assert np.array_equal(solve(inp), out)
                features[split].append(f)
        train = features["train"]
        for name in ("pure_both", "mixed_majority", "tie", "tie_color_contact", "opposing_rims", "diagonal", "adjacent", "local_shortcut_differs", "border_shortcut_differs"):
            assert any(f[name] for f in train), "Missing training witness: " + name
        assert any(f["extras"] for f in train), "Training must expose speck cleanup"
        # Neither a uniform halo nor independent per-pixel complements can fit.
        for color in (0, *train[0]["fills"]):
            assert any(not np.array_equal(np.where(_dilate(f["holes_mask"]) & ~f["holes_mask"], color,
                                                   np.where(f["holes_mask"], 0, f["clean"])), f["expected"])
                       for f in train)
        assert not any(f["clipped"] for f in features["test"]) or any(f["clipped"] for f in train)
        assert all(f["largest"] >= 7 and f["mixed_majority"] for f in features["test"])
        assert all(f["local_shortcut_differs"] for f in features["test"])
        assert max(f["largest"] for f in train) <= 5
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise AssertionError(str(exc)) from exc


def _canonical_structure(a):
    """Crop margins and normalize both material names and planar symmetries."""
    points = np.argwhere(a != 0)
    r, c = points.min(axis=0)
    rr, cc = points.max(axis=0)+1
    cropped = a[r:rr, c:cc]
    signatures = []
    for swapped in (False, True):
        renamed = np.where(cropped != 0, 3-cropped, 0) if swapped else cropped
        for turns in range(4):
            for reflected in (False, True):
                b = np.rot90(np.fliplr(renamed) if reflected else renamed, turns)
                signatures.append(_signature(b))
    return min(signatures)


def _semantic_regressions():
    """Fixed, manually checked rule examples, never consulted by solve()."""
    clean = np.zeros((18, 23), dtype=np.int64)
    clean[:16, 1:15] = 2
    clean[3:, 10:22] = 8
    points = [(0, 4), (6, 8), (6, 9), (6, 10), (10, 9), (10, 10),
              (11, 8), (14, 4), (14, 5), (14, 9), (15, 10)]
    inp = clean.copy()
    for p in points:
        inp[p] = 0
    inp[17, 1] = 8
    # This literal answer was checked as five distinct configurations: an edge
    # singleton, a 2:1 crack, a tied dimer beside a separate pure singleton,
    # a pure dimer, and two diagonal opposing singletons.
    rows = (
        "02280822222222200000000",
        "02288822222222200000000",
        "02222222222222200000000",
        "02222222228888888888880",
        "02222222228888888888880",
        "02222228888888888888880",
        "02222228000888888888880",
        "02222228888888888888880",
        "02222222228888888888880",
        "02222222000088888888880",
        "02222228000088888888880",
        "02222228000088888888880",
        "02222228888888888888880",
        "02288882888888888888880",
        "02280082800288888888880",
        "02288882800288888888880",
        "00000000022288888888880",
        "00000000008888888888880",
    )
    expected = np.array([[int(c) for c in row] for row in rows])
    assert np.array_equal(_recover_scene(inp)[0], clean)
    assert np.array_equal(solve(inp), expected)
    # Check both materials' role permutations, including numerical reversal.
    for first, second in ((1, 9), (9, 1), (8, 2)):
        rename = np.arange(10)
        rename[2], rename[8] = first, second
        for turns in range(4):
            for reflection in (False, True):
                def transform(a):
                    b = rename[a]
                    return np.rot90(np.fliplr(b) if reflection else b, turns)
                assert np.array_equal(solve(transform(inp)), transform(expected))
    # The solver supports compatible cracks beyond the sampler's 7/9 sizes.
    # Eleven horizontal holes erase five 2s and six 8s: the full rim is 2.
    long_input = clean.copy()
    long_input[6, 5:16] = 0
    long_expected = clean.copy()
    long_expected[5:8, 4:17] = 2
    long_expected[6, 5:16] = 0
    assert np.array_equal(solve(long_input), long_expected)
    # Removing the far-right vote changes the ballot to a tie, and changes
    # the far-left rim. This isolates the nonlocal interaction of A and B.
    tied_input = long_input.copy()
    tied_input[6, 15] = 8
    tied_expected = clean.copy()
    tied_expected[5:8, 4:16] = 0
    assert np.array_equal(solve(tied_input), tied_expected)
    assert solve(long_input)[5, 4] != solve(tied_input)[5, 4]
    # With the whole overlap erased, both occlusion orders fit. Reject it.
    ambiguous = np.zeros((18, 18), dtype=np.int64)
    ambiguous[1:10, 1:10] = 2
    ambiguous[8:17, 8:17] = 8
    ambiguous[8:10, 8:10] = 0
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous visible occlusion accepted")


def self_test():
    """Bounded author-owned checks: 42 seeds, invariants, symmetry, counterrules."""
    _semantic_regressions()
    shapes, clean_structures, palettes, sizes, counts, flags = set(), set(), set(), set(), set(), set()
    canonical_scenes, crack_structures = set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(pair["input"].shape)
                normalized = np.where(f["clean"] == f["fills"][0], 1, np.where(f["clean"] == f["fills"][1], 2, 0))
                clean_structures.add(_signature(normalized))
                canonical_scenes.add(_canonical_structure(normalized))
                for points, _, _, _ in f["ballots"]:
                    crack_structures.add(_canonical_structure(
                        np.where(_mask(normalized.shape, points), normalized, 0)))
                palettes.add(f["fills"])
                counts.add(len(f["boxes"]))
                sizes.add(f["holes"])
                flags.update(k for k in ("clipped", "branching", "tie", "tie_color_contact", "opposing_rims", "border_shortcut_differs") if f[k])
                assert np.array_equal(_constructed_output(f["clean"], f["holes_mask"]), pair["output"])
        if seed < 3:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], replay[s]) for k in a)
            # All eight planar symmetries preserve the rule, including clipping.
            for pair in puzzle["train"] + puzzle["test"]:
                for k in range(4):
                    for reflect in (False, True):
                        inp = np.rot90(np.fliplr(pair["input"]) if reflect else pair["input"], k)
                        out = np.rot90(np.fliplr(pair["output"]) if reflect else pair["output"], k)
                        assert np.array_equal(solve(inp), out)
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(clean_structures) >= 100
    assert len(canonical_scenes) >= 100 and len(crack_structures) >= 20
    assert len(palettes) >= 20 and len(sizes) >= 12
    assert counts == {2, 3, 4}
    assert flags == {"clipped", "branching", "tie", "tie_color_contact", "opposing_rims", "border_shortcut_differs"}
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(9137+a*10+b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mutated output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "clean_structures": len(clean_structures), "fill_palettes": len(palettes),
            "canonical_clean_scenes": len(canonical_scenes),
            "canonical_crack_structures": len(crack_structures),
            "hole_counts": len(sizes), "rectangle_counts": sorted(counts),
            "semantic_verification": "author-owned checks only"}
