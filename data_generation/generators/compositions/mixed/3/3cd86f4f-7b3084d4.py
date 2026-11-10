"""Marker-anchored four-piece assembly followed by a bottom-fixed row shear.

No parent imports, generated-example cache, or filesystem access is needed.
All randomness belongs to generate_puzzle's local NumPy generator.
"""
from __future__ import annotations

from hashlib import sha256
from math import isqrt

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
           "Grids must have two dimensions in 1..30.")
    _check(np.issubdtype(a.dtype, np.integer) and
           bool(np.all((a >= 0) & (a <= 9))), "Grid cells must be integers in 0..9.")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _tight(mask, colors):
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
    return np.where(mask[r0:r1, c0:c1], colors[r0:r1, c0:c1], -1).astype(np.int8)


def _components(mask, neighbors=NEIGHBORS):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r+dr, c+dc
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    cells.append(q)
        groups.append(cells)
    return groups


def _connected(mask, neighbors=NEIGHBORS):
    return len(_components(mask, neighbors)) == 1


def _rotations(piece):
    found, seen = [], set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            found.append(p)
    return found


def _shape_rotations(piece):
    return len(_rotations(np.where(piece >= 0, 1, -1).astype(np.int8)))


def _read_input(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _check(len(colors) == 6, "Expected background, four carriers, and one marker.")
    _check(int(np.sum(counts == counts.max())) == 1, "Background frequency must be untied.")
    background = int(colors[int(counts.argmax())])
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _check(len(singletons) == 1, "The marker is the only singleton color.")
    marker = singletons[0]
    foreground = a != background
    side = isqrt(int(foreground.sum()))
    _check(4 <= side <= 15 and side*side == int(foreground.sum()),
           "Foreground must have square area with side 4..15.")
    groups = _components(foreground)
    _check(len(groups) == 4, "Four pieces must be separated, including diagonally.")
    pieces, carriers = [], []
    for cells in groups:
        mask = np.zeros(a.shape, dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        p = _tight(mask, a)
        carrier = set(map(int, p[p >= 0])) - {marker}
        _check(len(cells) >= 3 and len(carrier) == 1,
               "Each piece has at least three cells and one carrier color.")
        pieces.append(p)
        carriers.append(carrier.pop())
    _check(len(set(carriers)) == 4, "Piece carriers must be distinct.")
    _check(sum(bool(np.any(p == marker)) for p in pieces) == 1,
           "Marker must belong to one piece.")
    return a, background, marker, side, pieces, carriers


def _assemblies(pieces, side, marker, limit=2):
    """Enumerate distinct colored covers, stopping once ambiguity is proved.

    A monochrome piece has the same colored placement for duplicate geometric
    rotations. The marked piece always puts its sole marker at board index 0.
    Thus a placement mask per distinct carrier fully identifies the answer.
    No generation metadata or proposed answer is consulted here.
    """
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    candidates, covering = [], []
    for i, piece in enumerate(pieces):
        unique = {}
        for p in _rotations(piece):
            h, w = p.shape
            if h > side or w > side:
                continue
            cells = tuple((int(r)*side+int(c), int(p[r, c]))
                          for r, c in np.argwhere(p >= 0))
            base = sum(1 << pos for pos, _ in cells)
            if i == anchor:
                offsets = [0] if p[0, 0] == marker else []
            else:
                offsets = [r*side+c for r in range(side-h+1) for c in range(side-w+1)]
            for offset in offsets:
                mask = base << offset
                unique.setdefault(mask, (mask, offset, cells))
        choices = list(unique.values())
        candidates.append(choices)
        by_cell = [[] for _ in range(side*side)]
        for candidate in choices:
            bits = candidate[0]
            while bits:
                bit = bits & -bits
                by_cell[bit.bit_length()-1].append(candidate)
                bits ^= bit
        covering.append(by_cell)
    if any(not c for c in candidates):
        return []
    full = (1 << (side*side))-1
    placed, answers, signatures = [None]*4, [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == full:
                signature = tuple(c[0] for c in placed)
                if signature not in signatures:
                    signatures.add(signature)
                    out = np.empty((side, side), dtype=np.int8)
                    for _, offset, cells in placed:
                        for pos, color in cells:
                            out.flat[pos+offset] = color
                    answers.append(out)
            return
        empty = full ^ occupied
        cell = (empty & -empty).bit_length()-1
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in covering[i][cell]:
                if not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    rest = tuple(i for i in range(4) if i != anchor)
    for candidate in candidates[anchor]:
        placed[anchor] = candidate
        search(candidate[0], rest)
        if len(answers) >= limit:
            break
    return answers


def _shear(square, background):
    n = square.shape[0]
    out = np.full((n, 2*n-1), background, dtype=np.int8)
    for r in range(n):
        out[r, n-1-r:2*n-1-r] = square[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use visible input only; reject missing or ambiguous anchored assemblies."""
    _, background, marker, side, pieces, _ = _read_input(input_grid)
    answers = _assemblies(pieces, side, marker)
    _check(len(answers) == 1, "Input must have one colored marker-anchored square assembly.")
    return _shear(answers[0], background)


def _growth(rng, side, diagonal, balanced):
    neighbors = NEIGHBORS if diagonal else EDGE
    labels = np.full((side, side), -1, dtype=np.int8)
    corners = [0, side-1, side*(side-1), side*side-1]
    starts = corners if balanced else [0]+list(map(int, rng.choice(
        np.arange(1, side*side), 3, replace=False)))
    weights = rng.uniform(.8, 1.3, 4) if balanced else np.exp(rng.uniform(-1.2, 1.2, 4))
    adjacency = [[(r+dr)*side+c+dc for dr, dc in neighbors
                  if 0 <= r+dr < side and 0 <= c+dc < side]
                 for r in range(side) for c in range(side)]
    frontier, sizes = [set() for _ in range(4)], [1]*4
    for i, pos in enumerate(starts):
        labels.flat[pos] = i
    for i, pos in enumerate(starts):
        frontier[i].update(q for q in adjacency[pos] if labels.flat[q] < 0)
    for _ in range(side*side-4):
        eligible = [i for i in range(4) if frontier[i]]
        # Establish the visible carrier/marker roles before unequal growth.
        # On a side-four board, unbounded weighted competition otherwise
        # routinely starves one piece below the three-cell minimum.
        small = [i for i in eligible if sizes[i] < 3]
        i = (min(small, key=lambda j: sizes[j]+float(rng.random())) if small else
             min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.5))
        choices = sorted(frontier[i])
        if diagonal and rng.random() < .75:
            edge = [q for q in choices if any(
                0 <= (r := q//side+dr) < side and
                0 <= (c := q % side+dc) < side and labels[r, c] == i
                for dr, dc in EDGE)]
            if edge:
                choices = edge
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(pos)
        frontier[i].update(q for q in adjacency[pos] if labels.flat[q] < 0)
    return labels


def _stepped_cuts(rng, side, diagonal):
    """Independent transverse cuts, then connected local interface transfers."""
    split = int(rng.integers(1, side))
    left, right = map(int, rng.integers(1, side, size=2))
    labels = np.empty((side, side), dtype=np.int8)
    labels[:left, :split], labels[left:, :split] = 0, 1
    labels[:right, split:], labels[right:, split:] = 2, 3
    for _ in range(side*3):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < side and 0 <= nc < side) or (r, c) == (0, 0):
            continue
        a, b = int(labels[r, c]), int(labels[nr, nc])
        if a == b or np.sum(labels == a) <= 3:
            continue
        labels[r, c] = b
        if not _connected(labels == a, NEIGHBORS if diagonal else EDGE):
            labels[r, c] = a
    return np.rot90(labels, int(rng.integers(4))).copy()


def _inserts(rng, side):
    """Three varied rectangular/L inserts carved from one connected wrapper."""
    labels = np.zeros((side, side), dtype=np.int8)
    for i in range(1, 4):
        for _ in range(40):
            h, w = map(int, rng.integers(1, max(3, side//2+1), size=2))
            if h*w < 3:
                continue
            r, c = int(rng.integers(side-h+1)), int(rng.integers(side-w+1))
            if (r, c) == (0, 0) or np.any(labels[r:r+h, c:c+w]):
                continue
            mask = np.ones((h, w), dtype=bool)
            if min(h, w) >= 2 and h*w > 4 and rng.random() < .6:
                mask[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
            if mask.sum() < 3:
                continue
            candidate = labels.copy()
            candidate[r:r+h, c:c+w][mask] = i
            if _connected(candidate == 0, EDGE):
                labels = candidate
                break
        else:
            return None
    return labels


def _colored_pieces(labels, carriers, marker):
    anchor = int(labels[0, 0])
    order = [anchor]+[i for i in range(4) if i != anchor]
    square = np.empty(labels.shape, dtype=np.int8)
    for label, color in zip(order, carriers):
        square[labels == label] = color
    square[0, 0] = marker
    return square, [_tight(labels == label, square) for label in order]


def _scatter(rng, pieces, background, anchor_rotation, loose):
    rotations = [anchor_rotation]+list(map(int, rng.integers(4, size=3)))
    # Ensure a clearly asymmetric rigid object actually needs a rotation.
    for i, p in enumerate(pieces):
        if _shape_rotations(p) == 4:
            if i != 0:
                rotations[i] = int(rng.integers(1, 4))
            break
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    full = (1 << 900)-1
    first_col = sum(1 << (r*30) for r in range(30))
    last_col = first_col << 29
    for attempt in range(10):
        order = sorted(range(4), key=lambda i: -oriented[i].size) if attempt == 0 else list(map(int, rng.permutation(4)))
        canvas = np.full((30, 30), -1, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        aspect = float(rng.uniform(.65, 1.5))
        for i in order:
            p = oriented[i]
            h, w = p.shape
            cells = [(int(r), int(c)) for r, c in np.argwhere(p >= 0)]
            base = sum(1 << (r*30+c) for r, c in cells)
            options = []
            for r in range(31-h):
                for c in range(31-w):
                    mask = base << (r*30+c)
                    if mask & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    score = (float(rng.random()) if loose else
                             hr*wc+abs(hr-aspect*wc)*2+float(rng.random())*20)
                    options.append((score, r, c, mask))
            if not options:
                break
            _, r, c, mask = min(options)
            horizontal = mask | ((mask & ~first_col) >> 1) | ((mask & ~last_col) << 1)
            blocked |= (horizontal | (horizontal << 30) | (horizontal >> 30)) & full
            view = canvas[r:r+h, c:c+w]
            view[p >= 0] = p[p >= 0]
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        else:
            rr, cc = np.nonzero(canvas >= 0)
            crop = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
            h, w = crop.shape
            H = h+int(rng.integers(min(5, 30-h)+1))
            W = w+int(rng.integers(min(5, 30-w)+1))
            area = int(np.sum(crop >= 0))
            largest = max(int(np.sum(crop == color)) for color in np.unique(crop) if color >= 0)
            while H*W-area <= largest:
                if H < 30 and (H <= W or W == 30):
                    H += 1
                elif W < 30:
                    W += 1
                else:
                    break
            if H*W-area <= largest:
                continue
            out = np.full((H, W), background, dtype=np.int8)
            r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
            out[r:r+h, c:c+w] = np.where(crop >= 0, crop, background)
            return out
    return None


def _make_pair(rng, side, background, marker, carriers, rotation,
               style, diagonal, must_diagonal, seen):
    for attempt in range(220):
        method = style if attempt < 110 else ("balanced" if side <= 5 else "growth")
        if method == "cuts":
            labels = _stepped_cuts(rng, side, diagonal)
        elif method == "inserts":
            labels = _inserts(rng, side)
        else:
            labels = _growth(rng, side, diagonal, method == "balanced")
        if labels is None or any(int(np.sum(labels == i)) < 3 for i in range(4)):
            continue
        square, pieces = _colored_pieces(labels, carriers, marker)
        if min(pieces[0].shape) < 2 or not all(_connected(p >= 0) for p in pieces):
            continue
        diagonal_present = any(not _connected(p >= 0, EDGE) for p in pieces)
        if must_diagonal and not diagonal_present:
            continue
        if any(len(_components(p >= 0, EDGE)) > 3 for p in pieces):
            continue
        if not any(_shape_rotations(p) == 4 for p in pieces):
            continue
        answers = _assemblies(pieces, side, marker)
        if len(answers) != 1 or not np.array_equal(answers[0], square):
            continue
        inp = _scatter(rng, pieces, background, rotation, bool(rng.random() < .18))
        if inp is not None and _key(inp) not in seen:
            seen.add(_key(inp))
            return {"input": inp, "output": _shear(square, background)}
    raise RuntimeError("Bounded construction exhausted; no ambiguous puzzle is returned.")


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range.")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    _check(a > b, "num_train must exceed num_test.")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    marker, fixed_background = palette[:2]
    vary_background = bool(rng.random() < .45)
    stable_carriers = bool(rng.random() < .25) and not vary_background
    diagonal_mode = bool(rng.random() < .4)
    larger = bool(rng.random() < .65)
    held = int(rng.integers(4))
    train_rotations = [k for k in range(4) if k != held] if not larger else [0, 1, 2, 3]
    # The first example always requires turning a two-dimensional marked piece.
    first_rotation = int(rng.choice([k for k in train_rotations if k != 0]))
    second_rotation = int(rng.choice([k for k in train_rotations if k != first_rotation]))
    rotations = [first_rotation, second_rotation]+[int(rng.choice(train_rotations)) for _ in range(a-2)]
    rotations += [int(rng.integers(4)) if larger else held for _ in range(b)]
    pool = list(range(4, 11 if larger else 13))
    sides = list(map(int, rng.permutation(pool)))
    while len(sides) < a:
        sides.extend(map(int, rng.permutation(pool)))
    sides = sides[:a]
    sides += [int(rng.integers(11, 16)) for _ in range(b)] if larger else [sides[0]]+[int(rng.integers(4, 16)) for _ in range(b-1)]
    backgrounds = [fixed_background]*(a+b)
    if vary_background:
        pool = [c for c in range(10) if c != marker]
        backgrounds = [int(c) for _ in range((a+b+8)//9) for c in rng.permutation(pool)][:a+b]
    styles = [str(s) for _ in range((a+b+3)//4)
              for s in rng.permutation(["balanced", "growth", "cuts", "inserts"])]
    result, seen, diagonal_seen = {"train": [], "test": []}, set(), False
    previous_anchor = None
    for i in range(a+b):
        background = backgrounds[i]
        if stable_carriers:
            carriers = palette[2:6]
        else:
            available = [c for c in range(10) if c not in (background, marker)]
            anchor = int(rng.choice([c for c in available if c != previous_anchor]))
            carriers = [anchor]+list(map(int, rng.choice([c for c in available if c != anchor], 3, replace=False)))
        previous_anchor = carriers[0]
        must_diagonal = diagonal_mode and i == 0
        diagonal = diagonal_mode and (i < a or diagonal_seen)
        style = "growth" if must_diagonal else styles[i]
        pair = _make_pair(rng, sides[i], background, marker, carriers, rotations[i],
                          style, diagonal, must_diagonal, seen)
        split = "train" if i < a else "test"
        result[split].append(pair)
        if i < a:
            pieces = _read_input(pair["input"])[4]
            diagonal_seen |= any(not _connected(p >= 0, EDGE) for p in pieces)
    return result


def _features(inp, out, background, marker, side, pieces, carriers):
    square = np.array([out[r, side-1-r:2*side-1-r] for r in range(side)], dtype=np.int8)
    anchor_carrier = next(c for p, c in zip(pieces, carriers) if np.any(p == marker))
    rotation, distortion, asymmetric_rotation = False, False, False
    areas = [int(np.sum(p >= 0)) for p in pieces]
    for p, c in zip(pieces, carriers):
        assembled = _tight((square == c) | ((square == marker) if c == anchor_carrier else False), square)
        moved = not np.array_equal(p, assembled)
        rotation |= moved
        asymmetric_rotation |= moved and _shape_rotations(p) == 4
        sheared = _tight((out == c) | ((out == marker) if c == anchor_carrier else False), out)
        distortion |= not any(np.array_equal(sheared, q) for q in _rotations(p))
    anchor = next(p for p in pieces if np.any(p == marker))
    r, c = map(int, np.argwhere(anchor == marker)[0])
    corner = (int(r == anchor.shape[0]-1), int(c == anchor.shape[1]-1))
    owners = square.copy()
    owners[owners == marker] = anchor_carrier
    normalized, roles = [], {}
    for v in owners.flat:
        normalized.append(roles.setdefault(int(v), len(roles)))
    return {"side": side, "background": background, "marker": marker,
            "anchor_carrier": anchor_carrier, "corner": corner,
            "diagonal": any(not _connected(p >= 0, EDGE) for p in pieces),
            "rotation": rotation, "asymmetric_rotation": asymmetric_rotation,
            "distortion": distortion, "area_ratio": max(areas)/min(areas),
            "min_area": min(areas), "symmetric": any(_shape_rotations(p) < 4 for p in pieces),
            "interior": any(not np.any(owners[0] == col) and not np.any(owners[-1] == col)
                            and not np.any(owners[:, 0] == col) and not np.any(owners[:, -1] == col)
                            for col in carriers),
            "corner_owners": len(set(map(int, owners[[0, 0, -1, -1], [0, -1, 0, -1]]))),
            "structure": sha256(bytes([side])+bytes(normalized)).hexdigest(),
            "density": side*side/inp.size}


def _validate_pair(pair):
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys.")
    inp, background, marker, side, pieces, carriers = _read_input(pair["input"])
    out = _grid(pair["output"])
    _check(out.shape == (side, 2*side-1), "Output dimensions must be n by 2n-1.")
    for r in range(side):
        start = side-1-r
        _check(np.all(out[r, :start] == background) and np.all(out[r, start+side:] == background),
               "Incorrect shear padding.")
        _check(np.all(out[r, start:start+side] != background), "Assembled rows must be gapless.")
    _check(int(out[0, side-1]) == marker, "Marker must shear from square's upper-left.")
    for color in range(10):
        if color != background:
            _check(int(np.sum(inp == color)) == int(np.sum(out == color)), "Colored-cell loss or recoloring.")
    _check(int(np.sum(out == background)) == side*(side-1), "Wrong padding area.")
    answers = _assemblies(pieces, side, marker)
    _check(len(answers) == 1, "Packing must have exactly one colored answer.")
    _check(np.array_equal(_shear(answers[0], background), out), "Wrong assembled row contents.")
    return _features(inp, out, background, marker, side, pieces, carriers)


def validate_generated_puzzle(puzzle):
    """Validate semantics and demonstration coverage, accepting arrays or lists."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys.")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Invalid pair counts.")
    facts, seen = {}, set()
    for split in ("train", "test"):
        facts[split] = []
        for pair in puzzle[split]:
            f = _validate_pair(pair)
            key = _key(np.asarray(pair["input"], dtype=np.int8))
            _check(key not in seen, "Inputs must be distinct within the puzzle.")
            seen.add(key)
            facts[split].append(f)
    train, test = facts["train"], facts["test"]
    _check(len({f["marker"] for f in train+test}) == 1, "Marker color must stay consistent.")
    _check(len({f["side"] for f in train}) >= 2, "Show at least two assembly sizes.")
    _check(any(f["asymmetric_rotation"] for f in train), "Demonstrate a required asymmetric rotation.")
    _check(any(f["distortion"] for f in train), "Demonstrate shear deforming a rigid piece.")
    for role in ("background", "anchor_carrier"):
        values = {f[role] for f in train}
        _check(len(values) > 1 or all(f[role] in values for f in test),
               "Show visible role variation before testing it.")
    _check(not any(f["diagonal"] for f in test) or any(f["diagonal"] for f in train),
           "Demonstrate diagonal attachments before testing them.")


def _signature(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """42 seeds, semantic ablations, metamorphic checks; no saved sweep payloads."""
    shapes, structures, signatures, sides, backgrounds, corners = set(), set(), set(), set(), set(), set()
    diagonal = symmetric = interior = uneven = 0
    larger = orientation_shift = pairs = 0
    for seed in range(42):
        a = 2+seed % 9
        b = 1+(seed//3) % min(4, a-1)
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        tr_sides, te_sides, tr_corners, te_corners = [], [], set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _validate_pair(pair)
                assert np.array_equal(solve(inp.copy()), out)
                shapes.add(inp.shape)
                structures.add(f["structure"])
                sides.add(f["side"])
                backgrounds.add(f["background"])
                corners.add(f["corner"])
                diagonal += f["diagonal"]
                symmetric += f["symmetric"]
                interior += f["interior"]
                uneven += f["area_ratio"] >= 4
                pairs += 1
                (tr_sides if split == "train" else te_sides).append(f["side"])
                (tr_corners if split == "train" else te_corners).add(f["corner"])
                # Omitting A yields an n by n square, never this output.
                assert out.shape[1] != out.shape[0]
                # Omitting B and shearing the scattered input uses the wrong row set.
                h, w = inp.shape
                assert (h, h+w-1) != out.shape
        larger += min(te_sides) > max(tr_sides)
        orientation_shift += bool(te_corners-tr_corners)
        if seed < 3:
            # Global recoloring changes visible roles but commutes with the rule.
            perm = np.roll(np.arange(10, dtype=np.int8), seed+1)
            p = puzzle["test"][0]
            assert np.array_equal(solve(perm[p["input"]]), perm[p["output"]])
            # Turning the scatter canvas changes piece poses, not the anchored answer.
            assert np.array_equal(solve(np.rot90(p["input"]).copy()), p["output"])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= pairs*.8
    assert sides == set(range(4, 16)) and backgrounds == set(range(10)) and len(corners) == 4
    assert diagonal >= 10 and symmetric >= 10 and uneven >= 20 and interior >= 5
    assert larger >= 10 and orientation_shift >= 10
    p = generate_puzzle(17, 4, 1)
    assert _signature(p) == _signature(generate_puzzle(17, 4, 1))
    assert _signature(p) != _signature(generate_puzzle(18, 4, 1))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(900+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    # Valid dimensions and color counts alone must not let a wrong rule pass.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = bad["test"][0]["output"]
    out[:] = out[:, ::-1].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Opposite shear direction accepted.")
    # Three interchangeable colored squares give distinct colored assemblies.
    # The solver must detect the ambiguity rather than choose a first packing.
    ambiguous = np.zeros((8, 8), dtype=np.int8)
    for i, (r, c) in enumerate(((0, 0), (0, 5), (5, 0), (5, 5)), 1):
        ambiguous[r:r+2, c:c+2] = i
    ambiguous[0, 0] = 5
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("Multiple colored packings accepted.")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes),
            "assembly_structures": len(structures), "diagonal_pairs": diagonal,
            "symmetric_pairs": symmetric, "interior_pairs": interior,
            "area_ratio_at_least_four": uneven, "larger_test_puzzles": larger,
            "unseen_marker_corner_puzzles": orientation_shift}
