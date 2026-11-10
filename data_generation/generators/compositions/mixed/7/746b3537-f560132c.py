"""Corner-keyed square assembly followed by adjacent full-vector compression.

No parent generator is imported. Construction expands random connected cuts;
solve segments and exhaustively packs only the supplied input pixels.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                other = (r + dr, c + dc)
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        result.append(np.asarray(sorted(cells), dtype=np.int16))
    return result


def _normal(coords):
    a = np.asarray(coords, dtype=np.int16)
    a = a - a.min(axis=0)
    return tuple(sorted(map(tuple, a.tolist())))


def _rotate(shape, turns):
    a = np.asarray(shape, dtype=np.int16)
    for _ in range(turns % 4):
        a = np.column_stack((a[:, 1], -a[:, 0]))
    return _normal(a)


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _mask(shape, side, row=0, col=0):
    return sum(1 << ((r + row) * side + c + col) for r, c in shape)


def _corner_options(shape, side, corner, rotations=range(4)):
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    options = {}
    for turns in rotations:
        oriented = _rotate(shape, turns)
        h, w = _dims(oriented)
        if h > side or w > side:
            continue
        r, c = (0 if corner < 2 else side - h), (0 if corner % 2 == 0 else side - w)
        mask = _mask(oriented, side, r, c)
        hits = tuple(bool(mask & (1 << i)) for i in corners)
        if sum(hits) == 1 and hits[corner]:
            options.setdefault(mask, turns)
    return tuple(options.items())


def _pack(shapes, side, anchor_corner=0, anchor_turn=0):
    """Return at most two distinct corner-labeled tilings, not piece permutations.

    A bounding box's offset is forced by the extreme corner it must occupy.
    Thus this finite search includes every allowed translation and rotation.
    """
    if len(shapes) != 4 or sum(map(len, shapes)) != side * side:
        return []
    anchor = _corner_options(shapes[0], side, anchor_corner, (anchor_turn,))
    if not anchor:
        return []
    remaining_corners = tuple(c for c in range(4) if c != anchor_corner)
    table = {(p, c): _corner_options(shapes[p], side, c) for p in range(1, 4) for c in remaining_corners}
    full = (1 << (side * side)) - 1
    masks, pieces, rotations = [None] * 4, [None] * 4, [None] * 4
    masks[anchor_corner], pieces[anchor_corner], rotations[anchor_corner] = anchor[0][0], 0, anchor_turn
    answers = {}

    def search(index, remaining, occupied):
        if len(answers) >= 2:
            return
        if index == len(remaining_corners):
            if occupied == full:
                sig = tuple(masks)
                record = {"masks": sig, "pieces": tuple(pieces), "rotations": tuple(rotations)}
                old = answers.get(sig)
                # Shape symmetry or interchangeable pieces must not inflate
                # the counted tilings or the measured rotation demand.
                score = lambda x: (sum(k != 0 for k in x["rotations"]), x["rotations"])
                if old is None or score(record) < score(old):
                    answers[sig] = record
            return
        corner = remaining_corners[index]
        for p in remaining:
            for mask, turns in table[p, corner]:
                if mask & occupied:
                    continue
                masks[corner], pieces[corner], rotations[corner] = mask, p, turns
                search(index + 1, tuple(q for q in remaining if q != p), occupied | mask)
    search(0, (1, 2, 3), anchor[0][0])
    return list(answers.values())


def _owner_from_solution(side, solution):
    owner = np.full((side, side), -1, dtype=np.int8)
    for corner, mask in enumerate(solution["masks"]):
        for i in range(side * side):
            if (mask >> i) & 1:
                owner.flat[i] = corner
    _require(bool(np.all(owner >= 0)), "packing has a gap")
    return owner


def _run_starts(a, axis):
    vectors = a if axis == 0 else a.T
    return np.asarray([0] + [i for i in range(1, len(vectors)) if not np.array_equal(vectors[i], vectors[i - 1])])


def _collapse(a):
    # Compare both axes on the complete packed image before deleting anything.
    return a[np.ix_(_run_starts(a, 0), _run_starts(a, 1))].copy()


def _returning(a, axis):
    vectors = a if axis == 0 else a.T
    rows = [tuple(map(int, v)) for v in vectors]
    reduced = [v for i, v in enumerate(rows) if i == 0 or v != rows[i - 1]]
    return len(set(reduced)) < len(reduced)


def _parse(input_grid):
    grid = _grid(input_grid)
    # The family has a uniform complete perimeter, making background visible
    # even if it is not globally the most frequent color.
    bg = int(grid[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "uniform background perimeter required")
    comps = _components(grid != bg)
    _require(len(comps) == 4, "exactly four foreground components required")
    anchors, plain = [], []
    for comp in comps:
        values = grid[comp[:, 0], comp[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) == 1:
            plain.append((comp, int(colors[0])))
            continue
        _require(len(colors) == 5 and sum(counts == 1) == 4 and sum(counts > 1) == 1, "keyed anchor must have four singleton key entries and a repeated carrier")
        points = comp[np.isin(values, colors[counts == 1])]
        r, c = map(int, points.min(axis=0))
        _require(set(map(tuple, points.tolist())) == {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}, "key must be 2x2")
        anchors.append((comp, grid[r:r + 2, c:c + 2].copy(), int(colors[counts > 1][0]), (r - int(comp[:, 0].min()), c - int(comp[:, 1].min()))))
    _require(len(anchors) == 1 and len(plain) == 3, "one anchor and three monochrome pieces required")
    anchor, key, carrier, key_location = anchors[0]
    ordered_comps = [anchor] + [p[0] for p in plain]
    shapes = [_normal(c) for c in ordered_comps]
    total = sum(map(len, shapes))
    side = math.isqrt(total)
    _require(side * side == total and 1 <= side <= 30, "total piece area must be a bounded square")
    for x, y in itertools.combinations(ordered_comps, 2):
        _require(int(np.max(np.abs(x[:, None, :] - y[None, :, :]), axis=2).min()) >= 2, "pieces require a background-cell gap")
    return {"grid": grid, "background": bg, "key": key, "shapes": shapes, "side": side,
            "components": ordered_comps, "carriers": (carrier,) + tuple(p[1] for p in plain), "key_location": key_location}


def _analyze(input_grid):
    info = _parse(input_grid)
    solutions = _pack(info["shapes"], info["side"])
    _require(len(solutions) == 1, "packing must have a unique corner-colored square before collapse")
    owner = _owner_from_solution(info["side"], solutions[0])
    packed = info["key"].reshape(-1)[owner]
    info.update(solution=solutions[0], owner=owner, packed=packed, output=_collapse(packed))
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels alone, with no generation state or examples."""
    return _analyze(input_grid)["output"].copy()


def _key_locations(owner):
    return [(r, c) for r in range(owner.shape[0] - 1) for c in range(owner.shape[1] - 1)
            if bool(np.all(owner[r:r + 2, c:c + 2] == 0))]


def _seamed(rng, h, w, recurrent=False):
    if recurrent:
        split = int(rng.integers(3, h - 2))
    else:
        split = int(rng.integers(1, h))
    owner = np.empty((h, w), dtype=np.int8)
    for lo, hi, left, right in ((0, split, 0, 1), (split, h, 2, 3)):
        cuts = rng.integers(1, w, size=hi - lo)
        if recurrent:
            a, b = rng.choice(np.arange(1, w), 2, replace=False)
            cuts[:3] = (a, b, a)
        for r, cut in zip(range(lo, hi), cuts):
            owner[r, :cut], owner[r, cut:] = left, right
    return owner


def _grown(rng, h, w):
    owner = np.full((h, w), -1, dtype=np.int8)
    frontier = []
    for i, (r, c) in enumerate(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        owner[r, c] = i
        frontier.extend((rr, cc, i) for dr, dc in _NEIGHBORS if 0 <= (rr := r + dr) < h and 0 <= (cc := c + dc) < w)
    # Unequal growth preferences generate small pieces and large wrapping pieces.
    weights = rng.uniform(0.3, 2.5, size=4)
    while frontier:
        probs = np.asarray([weights[t[2]] for t in frontier])
        index = int(rng.choice(len(frontier), p=probs / probs.sum()))
        r, c, label = frontier.pop(index)
        if owner[r, c] != -1:
            continue
        owner[r, c] = label
        frontier.extend((rr, cc, label) for dr, dc in _NEIGHBORS if 0 <= (rr := r + dr) < h and 0 <= (cc := c + dc) < w and owner[rr, cc] == -1)
    return owner


def _mutate(rng, owner):
    h, w = owner.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for _ in range(int(rng.integers(1, h * w // 2 + 1))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if (r, c) in corners:
            continue
        neighbors = [int(owner[r + dr, c + dc]) for dr, dc in _NEIGHBORS if 0 <= r + dr < h and 0 <= c + dc < w]
        old, new = int(owner[r, c]), int(rng.choice(neighbors))
        if old == new:
            continue
        owner[r, c] = new
        if any(len(_components(owner == k)) != 1 for k in (old, new)):
            owner[r, c] = old
    return owner


def _weights(rng, length, total):
    weights = np.ones(length, dtype=np.int16)
    # Include equal thickness when compatible; otherwise distribute independently.
    if total % length == 0 and rng.random() < 0.2:
        return np.full(length, total // length, dtype=np.int16)
    for i in rng.integers(0, length, size=total - length):
        weights[i] += 1
    return weights


def _geometry(rng, kind, target=False):
    """Bounded rejection, with explicit constraints checked after expansion."""
    for _ in range(500):
        if kind in ("row", "col"):
            h, w = int(rng.integers(6, 9)), int(rng.integers(4, 8))
            owner = _seamed(rng, h, w, recurrent=True)
            if rng.random() < 0.3:
                owner = _mutate(rng, owner)
            if kind == "col":
                owner = np.asarray([0, 2, 1, 3], dtype=np.int8)[owner.T]
        else:
            low, high = (5, 10) if target else (4, 8)
            h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
            owner = _grown(rng, h, w) if rng.random() < 0.45 else _seamed(rng, h, w)
            if rng.random() < 0.65:
                owner = _mutate(rng, owner)
        h, w = owner.shape
        side = int(rng.integers(max(h, w) + 2, 15))
        owner = np.repeat(np.repeat(owner, _weights(rng, h, side), axis=0), _weights(rng, w, side), axis=1)
        compact = _collapse(owner)
        if min(compact.shape) < 3 or not _key_locations(owner):
            continue
        longest = max(max(np.diff(np.r_[_run_starts(owner, axis), side])) for axis in (0, 1))
        # Hold out greater multiplicity, rather than an unseen operation or
        # compulsory simultaneous growth of all geometry/difficulty measures.
        if (target and longest < 5) or (not target and longest > 4):
            continue
        if kind == "row" and (not _returning(owner, 0) or compact.shape[0] == compact.shape[1]):
            continue
        if kind == "col" and not _returning(owner, 1):
            continue
        shapes = [_normal(np.argwhere(owner == k)) for k in range(4)]
        if min(map(len, shapes)) < 3 or len(shapes[0]) < 6:
            continue
        if kind in ("row", "col"):
            canonical = [min(_rotate(s, k) for k in range(4)) for s in shapes]
            if len(set(canonical)) != 4 or any(len({_rotate(s, k) for k in range(4)}) != 4 for s in shapes):
                continue
        solutions = _pack(shapes, side)
        expected = tuple(_mask(tuple(map(tuple, np.argwhere(owner == k).tolist())), side) for k in range(4))
        if len(solutions) == 1 and solutions[0]["masks"] == expected:
            return owner, shapes
    raise RuntimeError("bounded geometry search exhausted")


def _scatter(rng, shapes):
    """Free positions and shelf layouts; all inputs have a clear blank border."""
    dimensions = [_dims(s) for s in shapes]
    if rng.random() < 0.55:
        for _ in range(12):
            h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
            occupied = np.zeros((h, w), dtype=bool)
            offsets = [None] * 4
            for p in rng.permutation(4):
                ph, pw = dimensions[p]
                points = np.asarray(shapes[p])
                for _ in range(50):
                    r, c = int(rng.integers(1, h - ph)), int(rng.integers(1, w - pw))
                    rr, cc = points[:, 0] + r, points[:, 1] + c
                    if occupied[rr, cc].any():
                        continue
                    offsets[p] = (r, c)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            occupied[rr + dr, cc + dc] = True
                    break
                else:
                    break
            if all(o is not None for o in offsets):
                return offsets, (h, w)
    # Try row/column and 3+1/2+2 shelves in shuffled orders. At side <=14,
    # most two-shelf arrangements fit with a one-cell gap and blank border.
    layouts = []
    for transpose in (False, True):
        dims = [d[::-1] if transpose else d for d in dimensions]
        for _ in range(12):
            order = list(map(int, rng.permutation(4)))
            cuts = int(rng.integers(0, 8))
            groups, group = [], []
            for i, p in enumerate(order):
                group.append(p)
                if i == 3 or cuts & (1 << i):
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            heights = [max(dims[p][0] for p in g) for g in groups]
            widths = [sum(dims[p][1] for p in g) + gap * (len(g) - 1) for g in groups]
            bh, bw = sum(heights) + gap * (len(groups) - 1), max(widths)
            if bh > 28 or bw > 28:
                continue
            offsets, r = [None] * 4, 0
            for group, gh, gw in zip(groups, heights, widths):
                c = int(rng.integers(0, bw - gw + 1))
                for p in group:
                    rr = r + int(rng.integers(0, gh - dims[p][0] + 1))
                    offsets[p] = (c, rr) if transpose else (rr, c)
                    c += dims[p][1] + gap
                r += gh + gap
            layouts.append((offsets, (bw, bh) if transpose else (bh, bw)))
    if layouts:
        offsets, extent = layouts[int(rng.integers(len(layouts)))]
    else:
        # Try every two-shelf order and both axes with one-cell gaps. If
        # these do not fit, the caller samples a new geometry; no cropping.
        found = None
        for transpose in (False, True):
            dims = [d[::-1] if transpose else d for d in dimensions]
            for order in itertools.permutations(range(4)):
                a, b, c, d = order
                rh = max(dims[a][0], dims[b][0])
                bh = rh + 1 + max(dims[c][0], dims[d][0])
                bw = max(dims[a][1] + 1 + dims[b][1], dims[c][1] + 1 + dims[d][1])
                if bh <= 28 and bw <= 28:
                    offsets = [None] * 4
                    offsets[a], offsets[b] = (0, 0), (0, dims[a][1] + 1)
                    offsets[c], offsets[d] = (rh + 1, 0), (rh + 1, dims[c][1] + 1)
                    if transpose:
                        offsets = [(cc, rr) for rr, cc in offsets]
                        bh, bw = bw, bh
                    found = (offsets, (bh, bw))
                    break
            if found:
                break
        if found is None:
            return None
        offsets, extent = found
    pads = []
    for d in extent:
        extra = int(rng.integers(0, min(5, 28 - d) + 1))
        before = 1 + int(rng.integers(0, extra + 1))
        pads.append((before, 2 + extra - before))
    offsets = [(r + pads[0][0], c + pads[1][0]) for r, c in offsets]
    return offsets, tuple(extent[i] + sum(pads[i]) for i in range(2))


def _make_pair(rng, kind, key, target=False):
    for _ in range(20):
        owner, shapes = _geometry(rng, kind, target)
        turns = [0] + (list(map(int, rng.permutation([1, 2, 3]))) if kind == "row" else list(map(int, rng.integers(0, 4, 3))))
        if kind == "other" and rng.random() < 0.25:
            turns = [0, 0, 0, 0]
        oriented = [_rotate(s, k) for s, k in zip(shapes, turns)]
        scatter = _scatter(rng, oriented)
        if scatter is None:
            continue
        offsets, canvas = scatter
        if kind == "row" and sorted(range(4), key=lambda p: offsets[p]) == list(range(4)):
            # At least one demo must contradict input-reading-order coloring.
            continue
        available = [c for c in range(10) if c not in key]
        bg, carrier = map(int, rng.choice(available, 2, replace=False))
        if rng.random() < 0.4:
            carriers = [carrier] * 4
        else:
            carriers = [carrier] + list(map(int, rng.choice([c for c in range(10) if c != bg], 3)))
        grid = np.full(canvas, bg, dtype=np.int8)
        for p, shape in enumerate(oriented):
            coords = np.asarray(shape) + offsets[p]
            grid[coords[:, 0], coords[:, 1]] = carriers[p]
        kr, kc = _key_locations(owner)[int(rng.integers(len(_key_locations(owner))))]
        r, c = offsets[0]
        grid[r + kr:r + kr + 2, c + kc:c + kc + 2] = key.reshape(2, 2)
        output = _collapse(key[owner])
        return {"input": grid, "output": output.astype(np.int8)}
    raise RuntimeError("bounded scattering search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, count, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if not _integer(count) or not low <= count <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, seen_cuts = {"train": [], "test": []}, set(), set()
    keys = [rng.choice(10, 4, replace=False).astype(np.int8)]
    # A visible legend change is guaranteed even at minimum counts.
    second = keys[0].copy()
    index = int(rng.integers(4))
    second[index] = int(rng.choice([c for c in range(10) if c not in second]))
    keys.append(second)
    while len(keys) < num_train + num_test:
        mode = int(rng.integers(3))
        values = keys[int(rng.integers(len(keys)))].copy()
        if mode == 1:
            values = rng.choice(10, 4, replace=False).astype(np.int8)
        elif mode == 2:
            values = values[rng.permutation(4)]
        keys.append(values)
    for i, key in enumerate(keys):
        split = "train" if i < num_train else "test"
        kind = "row" if i == 0 else "col" if i == 1 else "other"
        for _ in range(30):
            pair = _make_pair(rng, kind, key, target=split == "test")
            signature = _signature(pair["input"])
            normalized = np.empty_like(pair["output"])
            for corner, color in enumerate(key):
                normalized[pair["output"] == color] = corner
            cut_signature = _signature(normalized)
            if signature not in seen and cut_signature not in seen_cuts:
                seen.add(signature)
                seen_cuts.add(cut_signature)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("distinct input search exhausted")
    # Demonstration order carries no semantics.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _local_collapse_differs(info):
    """A local piece repeat need not be a full packed-image repeat."""
    owner = info["owner"]
    for axis in (0, 1):
        full = owner if axis == 0 else owner.T
        for p in range(4):
            mask = full == p
            active = np.flatnonzero(mask.any(axis=1))
            for i, j in zip(active, active[1:]):
                if j == i + 1 and np.array_equal(mask[i], mask[j]) and not np.array_equal(full[i], full[j]):
                    return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema mismatch")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split must be a list")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen, cuts, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema mismatch")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            a = _analyze(inp)
            _require(np.array_equal(a["output"], out), "output differs from packing then collapse")
            cut_signature = _signature(_collapse(a["owner"]))
            _require(cut_signature not in cuts, "normalized compressed cuts must vary")
            cuts.add(cut_signature)
            n = a["side"]
            _require(3 <= out.shape[0] < n and 3 <= out.shape[1] < n, "both collapse axes must matter; no degenerate key-only mosaic")
            _require(set(map(int, out.flat)) == set(map(int, a["key"].flat)), "all four key colors must survive")
            _require(not np.array_equal(_collapse(inp), out), "omitting packing must change the answer")
            _require(np.array_equal(_collapse(out), out), "output must already have no adjacent repeated vectors")
            longest = max(max(np.diff(np.r_[_run_starts(a["packed"], axis), n])) for axis in (0, 1))
            _require(longest <= 4 if split == "train" else longest >= 5, "held-out run multiplicity schedule violated")
            # Pixel compression preserves each corner's color and all four pieces.
            _require(tuple(map(int, out[[0, 0, -1, -1], [0, -1, 0, -1]])) == tuple(map(int, a["key"].flat)), "corner key order lost")
            for color in a["key"].flat:
                _require(len(_components(out == color)) == 1, "piece disconnected by compression")
            if split == "train":
                training.append(a)
    _require(any(_returning(a["packed"], 0) for a in training), "training must retain a separated repeated row")
    _require(any(_returning(a["packed"], 1) for a in training), "training must retain a separated repeated column")
    _require(any(a["output"].shape[0] != a["output"].shape[1] for a in training), "rectangular compressed output demonstration required")
    _require(len({tuple(map(int, a["key"].flat)) for a in training}) >= 2, "training must change the visible legend")
    _require(any(_local_collapse_differs(a) for a in training), "training must distinguish global vectors from piecewise compression")
    _require(any(set(a["solution"]["rotations"]) == {0, 1, 2, 3} for a in training), "training must expose all quarter-turns")
    _require(any(tuple(a["solution"]["pieces"]) != tuple(sorted(range(4), key=lambda p: tuple(a["components"][p].min(axis=0)))) for a in training), "training must contradict scatter-reading-order coloring")


def measure_pair(pair):
    """Compact metrics derived from returned pixels, never latent metadata."""
    a = _analyze(pair["input"])
    owner, compact = a["owner"], _collapse(a["owner"])
    areas = [int(np.count_nonzero(owner == k)) for k in range(4)]
    straight = any(bool(np.all((owner[:r] < 2)) and np.all(owner[r:] >= 2)) for r in range(1, len(owner)))
    straight |= any(bool(np.all(np.isin(owner[:, :c], [0, 2])) and np.all(np.isin(owner[:, c:], [1, 3]))) for c in range(1, len(owner)))
    order = sorted(range(4), key=lambda p: tuple(a["components"][p].min(axis=0)))
    inverse = {p: c for c, p in enumerate(a["solution"]["pieces"])}
    return {"side": a["side"], "canvas": list(a["grid"].shape), "output": list(compact.shape),
            "row_runs": np.diff(np.r_[_run_starts(owner, 0), len(owner)]).tolist(),
            "column_runs": np.diff(np.r_[_run_starts(owner, 1), len(owner)]).tolist(),
            "returning_rows": _returning(owner, 0), "returning_columns": _returning(owner, 1),
            "areas": areas, "area_ratio": max(areas) / min(areas), "straight_split": straight,
            "rotations": list(a["solution"]["rotations"]), "background": a["background"],
            "carrier_count": len(set(a["carriers"])), "key": a["key"].reshape(-1).tolist(),
            "key_location": list(a["key_location"]), "reading_order": [inverse[p] for p in order],
            "local_collapse_differs": _local_collapse_differs(a),
            "normalized_output": hashlib.sha256(compact.tobytes() + bytes(compact.shape)).hexdigest()}


def self_test():
    """Stream 42 seeds; retain metrics and hashes only, no corpus or files."""
    fingerprints, canvases, outputs, cuts, orders, backgrounds, carriers = (set() for _ in range(7))
    returns = [0, 0]
    rotations, run_lengths = set(), set()
    nonstraight, unequal, rectangles = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                digest.update(pair["input"].tobytes())
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solve failed")
                # Column-first and row-first compression agree with independent
                # comparisons on the original square. No cascading differences.
                packed = a["packed"]
                sequential = packed[_run_starts(packed, 0)]
                sequential = sequential[:, _run_starts(sequential, 1)]
                _require(np.array_equal(sequential, a["output"]), "axis-order discrepancy")
                sequential = packed[:, _run_starts(packed, 1)]
                sequential = sequential[_run_starts(sequential, 0)]
                _require(np.array_equal(sequential, a["output"]), "column-first discrepancy")
                m = measure_pair(pair)
                canvases.add(tuple(m["canvas"])); outputs.add(tuple(m["output"]))
                cuts.add(m["normalized_output"]); orders.add(tuple(m["reading_order"]))
                backgrounds.add(m["background"]); carriers.add(m["carrier_count"])
                returns[0] += m["returning_rows"]; returns[1] += m["returning_columns"]
                rotations.update(m["rotations"])
                run_lengths.update(m["row_runs"] + m["column_runs"])
                nonstraight += not m["straight_split"]
                unequal += m["area_ratio"] >= 4
                rectangles += m["output"][0] != m["output"][1]
        fingerprints.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(puzzle[s][i][k], again[s][i][k]) for s in puzzle for i in range(len(puzzle[s])) for k in ("input", "output")), "determinism failed")
    _require(len(fingerprints) == 42, "seed diversity failed")
    _require(len(canvases) >= 35 and len(outputs) >= 12 and len(cuts) >= 100, "geometry variation insufficient")
    _require(len(orders) >= 15 and len(backgrounds) == 10 and carriers == {1, 2, 3, 4}, "scatter or palette role variation insufficient")
    _require(min(returns) >= 42 and rotations == {0, 1, 2, 3} and max(run_lengths) >= 4, "run recurrence/thickness/rotation variation missing")
    _require(nonstraight >= 10 and unequal >= 10 and rectangles >= 100, "structural variation missing")
    # Tiny analytic regression: separated equal vectors are retained.
    stripe = np.asarray([[1, 1, 2], [1, 1, 2], [3, 3, 4], [1, 1, 2]])
    _require(np.array_equal(_collapse(stripe), [[1, 2], [3, 4], [1, 2]]), "global deduplication replaced adjacent collapse")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
    damaged = generate_puzzle(17, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption accepted")
    pair = generate_puzzle(23, 2, 1)["test"][0]
    a = _analyze(pair["input"])
    recolored = pair["input"].copy()
    for comp in a["components"][1:]:
        recolored[comp[:, 0], comp[:, 1]] = a["carriers"][0]
    _require(np.array_equal(solve(recolored), pair["output"]), "plain carrier recoloring changed output")
    r, c = a["components"][0].min(axis=0) + a["key_location"]
    swapped = pair["input"].copy()
    swapped[r, c], swapped[r, c + 1] = swapped[r, c + 1], swapped[r, c]
    first, second = map(int, a["key"][0])
    expected = pair["output"].copy()
    expected[pair["output"] == first], expected[pair["output"] == second] = second, first
    _require(np.array_equal(solve(swapped), expected) and not np.array_equal(expected, pair["output"]), "key intervention failed")
