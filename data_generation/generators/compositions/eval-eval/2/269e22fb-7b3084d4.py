"""Snapshot jigsaws: a location-free photograph resolves an ambiguous rigid tiling.

Only Python/NumPy are required. No parent files, seeds, training labels, caches,
or generation metadata are consulted by solve(). No files are written here.
"""
from __future__ import annotations

from hashlib import sha256
from math import isqrt

import numpy as np


EDGE = ((-1, 0), (0, -1), (0, 1), (1, 0))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _tight(a):
    rr, cc = np.nonzero(a)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _connected(a, neighbors=EDGE):
    cells = set(map(tuple, np.argwhere(a)))
    if not cells:
        return False
    reached = {min(cells)}
    todo = list(reached)
    while todo:
        r, c = todo.pop()
        for dr, dc in neighbors:
            p = r+dr, c+dc
            if p in cells and p not in reached:
                reached.add(p)
                todo.append(p)
    return len(reached) == len(cells)


def _orientations(mask):
    seen = set()
    result = []
    for k in range(4):
        a = np.rot90(mask, k).copy()
        if _key(a) not in seen:
            seen.add(_key(a))
            result.append(a)
    return result


def _rotation_class(a):
    return min(_key(np.rot90(a, k)) for k in range(4))


def _components(a, background):
    remaining = set(map(tuple, np.argwhere(a != background)))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in NEAR:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        rr, cc = np.array(cells).T
        box = a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()
        mask = np.zeros(box.shape, dtype=bool)
        mask[rr-rr.min(), cc-cc.min()] = True
        objects.append((box, mask))
    return objects


def _read_input(input_grid):
    """The outer border identifies space; the sole framed component is the cue."""
    a = _grid(input_grid)
    background = int(a[0, 0])
    _require(all(np.all(edge == background) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "A complete outer background border is required")
    pieces, snapshots = [], []
    for box, mask in _components(a, background):
        colors = np.unique(box[mask])
        if len(colors) == 1:
            _require(int(mask.sum()) >= 3, "A piece needs at least three cells")
            pieces.append((int(colors[0]), mask))
            continue
        _require(mask.all() and min(box.shape) >= 4, "The only multicolor object is a framed rectangle")
        frame = int(box[0, 0])
        _require(all(np.all(edge == frame) for edge in (box[0], box[-1], box[:, 0], box[:, -1])),
                 "Snapshot must have a uniform complete frame")
        cue = box[1:-1, 1:-1].copy()
        _require(not np.any(cue == frame), "Frame color cannot appear inside the snapshot")
        snapshots.append((frame, cue))
    _require(len(snapshots) == 1 and 3 <= len(pieces) <= 5, "Need one snapshot and three to five pieces")
    frame, cue = snapshots[0]
    colors = [c for c, _ in pieces]
    _require(len(set(colors)) == len(colors) and frame not in colors, "Piece colors must be distinct from frame")
    _require(set(map(int, np.unique(cue))) <= set(colors) and len(np.unique(cue)) >= 3,
             "Snapshot must show at least three piece colors")
    area = sum(int(p.sum()) for _, p in pieces)
    side = isqrt(area)
    _require(side * side == area and 5 <= side <= 9, "Piece area must define a side-five-to-nine square")
    _require(all(d <= side for d in cue.shape) and cue.size < area, "Snapshot must be a proper crop")
    return a, background, frame, cue, pieces, side


def _placements(pieces, side):
    options = []
    for color, mask in pieces:
        by_mask = {}
        for p in _orientations(mask):
            h, w = p.shape
            if h > side or w > side:
                continue
            base = sum(1 << (int(r)*side+int(c)) for r, c in np.argwhere(p))
            for r in range(side-h+1):
                for c in range(side-w+1):
                    bits = base << (r*side+c)
                    by_mask[bits] = bits
        options.append(tuple(by_mask))
    return options


def _covers(pieces, side, options, limit=2, known=None):
    """Exact rigid covers. known restricts colors at selected target positions.

    Stop only after finding limit different colored results. There is one
    monochrome piece of each color, so placements and colored results coincide.
    """
    if known is not None:
        all_known = sum(known.values())
        options = [tuple(m for m in choices if m & known.get(color, 0) == known.get(color, 0)
                         and not m & (all_known ^ known.get(color, 0)))
                   for (color, _), choices in zip(pieces, options)]
    if any(not choices for choices in options):
        return []
    full = (1 << (side*side))-1
    chosen = [0]*len(pieces)
    results = []

    def search(occupied, remaining):
        if len(results) >= limit:
            return
        if not remaining:
            if occupied == full:
                out = np.empty((side, side), dtype=np.int8)
                for (color, _), bits in zip(pieces, chosen):
                    while bits:
                        bit = bits & -bits
                        out.flat[bit.bit_length()-1] = color
                        bits ^= bit
                results.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for bits in options[i]:
                if bits & first and not bits & occupied:
                    chosen[i] = bits
                    search(occupied | bits, rest)
                    if len(results) >= limit:
                        return

    search(0, tuple(range(len(pieces))))
    return results


def _snapshot_answers(pieces, side, cue, limit=2):
    options = _placements(pieces, side)
    answers = {}
    h, w = cue.shape
    for r in range(side-h+1):
        for c in range(side-w+1):
            known = {}
            for pr in range(h):
                for pc in range(w):
                    color = int(cue[pr, pc])
                    known[color] = known.get(color, 0) | (1 << ((r+pr)*side+c+pc))
            for out in _covers(pieces, side, options, limit=limit, known=known):
                answers[_key(out)] = out
                if len(answers) >= limit:
                    return list(answers.values())
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the sole rigid square containing the untransformed snapshot."""
    _, _, _, cue, pieces, side = _read_input(input_grid)
    answers = _snapshot_answers(pieces, side, cue)
    _require(len(answers) == 1, "Snapshot and pieces must determine exactly one answer")
    return answers[0].copy()


def _insert_shape(rng):
    kind = int(rng.integers(4))
    if kind == 0:
        return np.ones((1, int(rng.integers(3, 5))), dtype=bool)
    if kind == 1:
        return np.ones((2, int(rng.integers(2, 4))), dtype=bool)
    cells = {(0, 0)}
    for _ in range(int(rng.integers(2, 7))):
        choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in EDGE} - cells)
        choices = [p for p in choices if max(r for r, _ in cells | {p})-min(r for r, _ in cells | {p}) < 4
                   and max(c for _, c in cells | {p})-min(c for _, c in cells | {p}) < 4]
        cells.add(choices[int(rng.integers(len(choices)))])
    lo = np.min(list(cells), axis=0)
    hi = np.max(list(cells), axis=0)
    shape = np.zeros(tuple(hi-lo+1), dtype=bool)
    for r, c in cells:
        shape[r-lo[0], c-lo[1]] = True
    return shape


def _partition(rng, side, count):
    """Two congruent inserts plus a procedurally partitioned connected remainder."""
    labels = np.full((side, side), -1, dtype=np.int8)
    shape = _insert_shape(rng)
    for i in range(2):
        p = np.rot90(shape, int(rng.integers(4)))
        h, w = p.shape
        if h > side or w > side:
            return None
        choices = []
        for r in range(side-h+1):
            for c in range(side-w+1):
                if np.all(labels[r:r+h, c:c+w][p] < 0):
                    choices.append((r, c))
        if not choices:
            return None
        r, c = choices[int(rng.integers(len(choices)))]
        labels[r:r+h, c:c+w][p] = i
    available = labels < 0
    if not _connected(available) or available.sum() < 3*(count-2):
        return None
    seeds = rng.choice(np.flatnonzero(available), count-2, replace=False)
    weights = rng.uniform(.4, 2.0, count-2)
    for i, pos in enumerate(seeds, 2):
        labels.flat[int(pos)] = i
    while np.any(labels < 0):
        frontiers = []
        for i in range(2, count):
            frontier = sorted({(int(r)+dr, int(c)+dc) for r, c in np.argwhere(labels == i)
                               for dr, dc in EDGE if 0 <= r+dr < side and 0 <= c+dc < side
                               and labels[r+dr, c+dc] < 0})
            if frontier:
                frontiers.append((float(np.sum(labels == i))/weights[i-2]+float(rng.random()), i, frontier))
        if not frontiers:
            return None
        _, i, choices = min(frontiers)
        labels[choices[int(rng.integers(len(choices)))]] = i
    if min(int(np.sum(labels == i)) for i in range(count)) < 3:
        return None
    return labels


def _cue_locations(full, cue):
    h, w = cue.shape
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    return list(zip(*np.nonzero(np.all(windows == cue, axis=(-1, -2)))))


def _choose_cue(rng, target, assemblies, fraction):
    side = target.shape[0]
    choices = []
    for h in range(2, side+1):
        for w in range(2, side+1):
            area = h*w/(side*side)
            if fraction[0] <= area <= fraction[1] and area < 1:
                choices.append((h, w))
    rng.shuffle(choices)
    # Test requests supply a smaller admissible fraction band; trials are bounded.
    for h, w in choices:
        offsets = [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
        rng.shuffle(offsets)
        for r, c in offsets[:12]:
            cue = target[r:r+h, c:c+w].copy()
            if len(np.unique(cue)) < 3:
                continue
            # A seam-spanning photograph must actually identify a unique image
            # and unique location, including against rotated competing packings.
            if len(_cue_locations(target, cue)) != 1:
                continue
            if any(_cue_locations(out, cue) for out in assemblies if not np.array_equal(out, target)):
                continue
            return cue
    return None


def _scatter(rng, pieces, cue, background, frame):
    panel = np.pad(cue, 1, constant_values=frame)
    objects = [(panel, np.ones(panel.shape, dtype=bool))]
    rotation_needed = False
    for color, mask in pieces:
        k = int(rng.integers(1, 4))
        p = np.rot90(mask, k).copy()
        rotation_needed |= not np.array_equal(p, mask)
        objects.append((np.full(p.shape, color, dtype=np.int8), p))
    if not rotation_needed:
        return None
    for _ in range(6):
        canvas = np.full((28, 28), background, dtype=np.int8)
        blocked = np.zeros((28, 28), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: -objects[i][1].size+float(rng.uniform(-8, 8)))
        maxr = maxc = 0
        okay = True
        for i in order:
            paint, mask = objects[i]
            h, w = mask.shape
            rr, cc = np.nonzero(mask)
            candidates = []
            jitter = float(rng.uniform(4, 32))
            aspect = float(rng.uniform(.75, 1.3))
            for r in range(29-h):
                for c in range(29-w):
                    if not np.any(blocked[r+rr, c+cc]):
                        H, W = max(maxr, r+h), max(maxc, c+w)
                        candidates.append((H*W+abs(H-aspect*W)*2+float(rng.random())*jitter, r, c))
            if not candidates:
                okay = False
                break
            _, r, c = min(candidates)
            canvas[r+rr, c+cc] = paint[rr, cc]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    br, bc = r+rr+dr, c+cc+dc
                    keep = (br >= 0) & (br < 28) & (bc >= 0) & (bc < 28)
                    blocked[br[keep], bc[keep]] = True
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not okay:
            continue
        rr, cc = np.nonzero(canvas != background)
        canvas = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
        h, w = canvas.shape
        vertical = int(rng.integers(2, min(6, 30-h)+1))
        horizontal = int(rng.integers(2, min(6, 30-w)+1))
        top, left = int(rng.integers(1, vertical)), int(rng.integers(1, horizontal))
        canvas = np.pad(canvas, ((top, vertical-top), (left, horizontal-left)), constant_values=background)
        return canvas
    return None


def _make_pair(rng, side, count, colors, background, frame, fraction, seen, nonorigin=False):
    for _ in range(160):
        labels = _partition(rng, side, count)
        if labels is None:
            continue
        pieces = [(colors[i], _tight(labels == i)) for i in range(count)]
        if not any(len(_orientations(p)) == 4 for _, p in pieces):
            continue
        options = _placements(pieces, side)
        assemblies = _covers(pieces, side, options, limit=65)
        if len(assemblies) >= 65 or len({_rotation_class(a) for a in assemblies}) < 2:
            continue
        target = np.array(colors, dtype=np.int8)[labels]
        cue = _choose_cue(rng, target, assemblies, fraction)
        if cue is None:
            continue
        inp = _scatter(rng, pieces, cue, background, frame)
        if inp is not None:
            rotations = list(range(4))
            if nonorigin:
                rotations = [k for k in rotations if _cue_locations(np.rot90(target, k), np.rot90(cue, k))[0] != (0, 0)]
            k = int(rng.choice(rotations))
            inp, target = np.rot90(inp, k).copy(), np.rot90(target, k).copy()
        if inp is not None and _key(inp) not in seen:
            seen.add(_key(inp))
            return {"input": inp, "output": target}
    raise RuntimeError("Bounded snapshot-jigsaw construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    background, frame = palette[:2]
    base = int(rng.integers(5, 8))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt+nq):
        split = "train" if i < nt else "test"
        if i == 0:
            side, count, fraction = base, 3, (.48, .85)
        elif i == 1:
            side, count, fraction = min(8, base+1), 4, (.30, .60)
        elif split == "train":
            side, count, fraction = int(rng.integers(5, 9)), int(rng.integers(3, 6)), (.25, .75)
        else:
            side, count, fraction = int(rng.integers(max(7, base+1), 10)), int(rng.integers(3, 6)), (.10, .29)
        colors = list(map(int, rng.permutation(palette[2:])[:count]))
        puzzle[split].append(_make_pair(rng, side, count, colors, background, frame, fraction, seen, nonorigin=i == 0))
    return puzzle


def _deformation_witness(out, cue, pieces):
    """A concrete A-without-B counterexample, preserving the entire photograph.

    Swap two differently colored cells outside the photograph. Require that
    at least one resulting color mask is no longer a rigid supplied object.
    This also preserves the square size and every color's total cell count.
    """
    r, c = _cue_locations(out, cue)[0]
    outside = np.ones(out.shape, dtype=bool)
    outside[r:r+cue.shape[0], c:c+cue.shape[1]] = False
    positions = list(map(int, np.flatnonzero(outside)))
    for i, first in enumerate(positions):
        for second in positions[i+1:]:
            if out.flat[first] == out.flat[second]:
                continue
            changed = out.copy()
            changed.flat[first], changed.flat[second] = int(out.flat[second]), int(out.flat[first])
            if any(not any(np.array_equal(_tight(changed == color), p) for p in _orientations(mask))
                   for color, mask in pieces):
                return first, second
    return None


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Illegal counts")
    seen, conventions = set(), set()
    deformation_teaching = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            a, background, frame, cue, pieces, side = _read_input(pair["input"])
            out = _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            conventions.add((background, frame))
            _require(out.shape == (side, side), "Output side must follow piece area")
            _require(background not in out and frame not in out, "Only pieces enter the answer")
            _require(len(_cue_locations(out, cue)) == 1, "Snapshot needs a unique exact location")
            masks = []
            rotated_asymmetry = False
            for color, mask in pieces:
                actual = _tight(out == color) if np.any(out == color) else np.zeros((1, 1), dtype=bool)
                _require(np.sum(out == color) == mask.sum(), "Conserve every piece cell")
                _require(any(np.array_equal(actual, p) for p in _orientations(mask)), "Do not reflect or deform pieces")
                rotated_asymmetry |= len(_orientations(mask)) == 4 and not np.array_equal(actual, mask)
                masks.append(min(_key(p) for p in _orientations(mask)))
            _require(len(set(masks)) < len(masks), "Include congruent differently colored pieces")
            _require(rotated_asymmetry, "Demonstrate a visible rigid rotation")
            options = _placements(pieces, side)
            unconstrained = _covers(pieces, side, options, limit=65)
            _require(len(unconstrained) < 65, "Generation audit must enumerate all alternatives")
            _require(len({_rotation_class(g) for g in unconstrained}) >= 2,
                     "Snapshot must resolve more than global rotation")
            answers = _snapshot_answers(pieces, side, cue)
            _require(len(answers) == 1 and np.array_equal(answers[0], out), "Wrong or ambiguous joint answer")
            if split == "train":
                deformation_teaching |= _deformation_witness(out, cue, pieces) is not None
    _require(len(conventions) == 1, "Background/frame colors are stable within a puzzle")
    _require(deformation_teaching, "Training must defeat crop-only color-count completion")
    train_fractions = []
    test_fractions = []
    for split, fractions in (("train", train_fractions), ("test", test_fractions)):
        for pair in puzzle[split]:
            _, _, _, cue, _, side = _read_input(pair["input"])
            fractions.append(cue.size/(side*side))
    _require(max(test_fractions) < min(train_fractions[:2]), "Tests must show less context than the two teaching examples")


def _digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming audit: 42 seeds; keep hashes/feature sets, not corpora."""
    signatures, shapes, outputs, fractions, counts, cue_shapes, structures = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, _, _, cue, pieces, side = _read_input(pair["input"])
                shapes.add(a.shape)
                outputs.add(side)
                counts.add(len(pieces))
                cue_shapes.add(cue.shape)
                fractions.add(round(cue.size/(side*side), 3))
                mapping = {}
                normalized = np.array([mapping.setdefault(int(c), len(mapping)) for c in pair["output"].flat], dtype=np.int8)
                structures.add((side, normalized.tobytes()))
                _require(np.array_equal(solve(a), pair["output"]), "Input-only solve failed")
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "Nondeterminism")
    _require(len(signatures) == 42 and len(structures) >= 190, "Insufficient structural variation")
    _require(outputs == set(range(5, 10)) and counts == {3, 4, 5}, "Missing sizes/counts")
    _require(len(shapes) >= 35 and len(cue_shapes) >= 20 and len(fractions) >= 35, "Missing geometric/crop variation")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(71, nt, nq)
        _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "Counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = int(bad["test"][0]["input"][0, 0])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "output_sides": sorted(outputs),
            "piece_counts": sorted(counts), "input_shapes": len(shapes), "cue_shapes": len(cue_shapes),
            "normalized_outputs": len(structures)}
