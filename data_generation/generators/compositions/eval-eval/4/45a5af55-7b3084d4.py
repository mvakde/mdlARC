"""Broken growth rings: rigid assembly and concentric inpainting.

No files, retained examples, or generation metadata are used by solve().
Background is the input perimeter color; 5 is unpainted material and 9 is
the unique anchor. Other colors, including 0, are literal pigments.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
from math import isqrt

import numpy as np


NEUTRAL = 5
ANCHOR = 9
EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_), "integer grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _depths(side):
    d = np.minimum(np.arange(side), np.arange(side)[::-1])
    return np.minimum(d[:, None], d[None, :])


def _wrap(code, side):
    """Construct the intended painting from the center by border addition."""
    center = 1 if side % 2 else 2
    result = np.full((center, center), code[-1], dtype=np.int8)
    for color in code[-2::-1]:
        result = np.pad(result, 1, constant_values=int(color))
    result[0, 0] = ANCHOR
    return result


def _components(mask, neighbors=NEIGHBORS):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    cells.append(point)
        result.append(cells)
    return result


def _tight(a, cells):
    rr, cc = np.asarray(cells).T
    result = np.full((int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)), -1, dtype=np.int8)
    result[rr-rr.min(), cc-cc.min()] = a[rr, cc]
    return result


def _orientations(piece):
    seen = set()
    for rotation in range(4):
        oriented = np.rot90(piece, rotation).copy()
        key = _key(oriented)
        if key not in seen:
            seen.add(key)
            yield oriented


def _read_input(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(bg not in (NEUTRAL, ANCHOR), "background conflicts with material or anchor")
    _require(bool(np.all(a[0] == bg) and np.all(a[-1] == bg)
                  and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)), "uniform empty perimeter")
    _require(np.sum(a == ANCHOR) == 1, "one anchor required")
    area = int(np.sum(a != bg))
    side = isqrt(area)
    _require(side * side == area and 4 <= side <= 12, "foreground must form a square of side 4..12")
    groups = _components(a != bg)
    _require(3 <= len(groups) <= 5, "three to five separated pieces required")
    pieces = [_tight(a, cells) for cells in groups]
    for p in pieces:
        _require(np.sum(p >= 0) >= 3, "piece too small")
        _require(bool(np.any((p >= 0) & (p != NEUTRAL) & (p != ANCHOR))), "piece needs pigment")
    _require(bool(np.any(a == NEUTRAL)), "unpainted cells required")
    return a, bg, side, pieces


def _placements(pieces, side, radial=True):
    depths = _depths(side).ravel()
    candidates = []
    for piece in pieces:
        marked = bool(np.any(piece == ANCHOR))
        options = {}
        for p in _orientations(piece):
            h, w = p.shape
            if h > side or w > side:
                continue
            if marked and p[0, 0] != ANCHOR:
                continue
            rr, cc = np.nonzero(p >= 0)
            colors = tuple(map(int, p[rr, cc]))
            positions = tuple(int(r)*side+int(c) for r, c in zip(rr, cc))
            offsets = [0] if marked else [r*side+c for r in range(side-h+1) for c in range(side-w+1)]
            for offset in offsets:
                cells = tuple((pos+offset, color) for pos, color in zip(positions, colors))
                assignments = {}
                valid = True
                if radial:
                    for pos, color in cells:
                        if color in (NEUTRAL, ANCHOR):
                            continue
                        d = int(depths[pos])
                        if d in assignments and assignments[d] != color:
                            valid = False
                            break
                        assignments[d] = color
                if valid:
                    mask = sum(1 << pos for pos, _ in cells)
                    assignment = tuple(sorted(assignments.items()))
                    signature = (mask, assignment) if radial else (mask, cells)
                    options.setdefault(signature, (mask, assignment, cells))
        candidates.append(list(options.values()))
    return candidates


def _assemblies(pieces, side, radial=True, limit=2):
    """Exact cover coupled to depth-color unification; deduplicate by answer.

    A radial solution with an unsampled depth is ambiguous and is returned
    with -1 at that depth, causing solve to reject it. No search cutoff.
    """
    options = _placements(pieces, side, radial)
    if any(not candidates for candidates in options):
        return []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == ANCHOR))
    full = (1 << (side*side))-1
    placed = [None] * len(pieces)
    answers = {}
    visited = set()
    layers = (side+1)//2

    def search(occupied, remaining, code):
        if len(answers) >= limit:
            return
        # Equivalent states can arise from symmetric or interchangeable pieces.
        state = occupied, remaining, code
        if radial and state in visited:
            return
        if radial:
            visited.add(state)
        if not remaining:
            if occupied == full:
                raw = np.full((side, side), -1, dtype=np.int8)
                for candidate in placed:
                    for pos, color in candidate[2]:
                        raw.flat[pos] = color
                signature = code if radial else raw.tobytes()
                answers.setdefault(signature, (code, raw))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        eligible = []
        for i in remaining:
            fitting = []
            for candidate in options[i]:
                mask, assignments, _ = candidate
                if mask & occupied:
                    continue
                if any(code[d] >= 0 and code[d] != c for d, c in assignments):
                    continue
                fitting.append(candidate)
            if not fitting:
                return
            eligible.append((i, fitting))
        eligible.sort(key=lambda item: len(item[1]))
        for i, fitting in eligible:
            rest = tuple(j for j in remaining if j != i)
            for candidate in fitting:
                if not candidate[0] & first:
                    continue
                updated = list(code)
                for d, color in candidate[1]:
                    updated[d] = color
                placed[i] = candidate
                search(occupied | candidate[0], rest, tuple(updated))
                if len(answers) >= limit:
                    return

    for candidate in options[anchor]:
        code = [-1] * layers
        for d, color in candidate[1]:
            code[d] = color
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(pieces)) if i != anchor), tuple(code))
        if len(answers) >= limit:
            break
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the given grid; reject inconsistent or ambiguous assemblies."""
    _, _, side, pieces = _read_input(input_grid)
    answers = _assemblies(pieces, side)
    _require(len(answers) == 1 and -1 not in answers[0][0], "painting is not uniquely determined")
    return _wrap(answers[0][0], side)


def _partition(rng, side, count, style):
    labels = np.full((side, side), -1, dtype=np.int8)
    if style == "slabs":
        cuts = sorted(map(int, rng.choice(np.arange(1, side), count-1, replace=False)))
        for i, (a, b) in enumerate(zip([0]+cuts, cuts+[side])):
            labels[:, a:b] = i
        if rng.random() < .5:
            labels = labels.T.copy()
        return labels
    if style == "cuts":
        boxes = [(0, 0, side, side)]
        while len(boxes) < count:
            possibilities = []
            for i, (r, c, h, w) in enumerate(boxes):
                possibilities.extend((i, 0, k) for k in range(1, h) if min(k, h-k)*w >= 3)
                possibilities.extend((i, 1, k) for k in range(1, w) if min(k, w-k)*h >= 3)
            if not possibilities:
                return None
            i, axis, k = possibilities[int(rng.integers(len(possibilities)))]
            r, c, h, w = boxes.pop(i)
            boxes.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                         [(r, c, h, k), (r, c+k, h, w-k)])
        for i, (r, c, h, w) in enumerate(boxes):
            labels[r:r+h, c:c+w] = i
        return labels
    diagonal = style == "diagonal"
    neighbors = NEIGHBORS if diagonal else EDGE
    starts = list(map(int, rng.choice(side*side, count, replace=False)))
    frontiers = [set() for _ in range(count)]
    weights = np.exp(rng.uniform(-.8, .8, count))
    sizes = [1]*count

    def adjacent(pos):
        r, c = divmod(pos, side)
        return [rr*side+cc for dr, dc in neighbors
                if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]

    for i, start in enumerate(starts):
        labels.flat[start] = i
    for i, start in enumerate(starts):
        frontiers[i].update(p for p in adjacent(start) if labels.flat[p] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if frontiers[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.6)
        choices = sorted(frontiers[i])
        if diagonal and rng.random() < .8:
            edge_choices = [p for p in choices if any(0 <= (r := p//side+dr) < side and
                             0 <= (c := p%side+dc) < side and labels[r, c] == i for dr, dc in EDGE)]
            choices = edge_choices or choices
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontiers:
            f.discard(pos)
        frontiers[i].update(p for p in adjacent(pos) if labels.flat[p] < 0)
    return labels


def _scatter(rng, pieces, bg):
    oriented = [np.rot90(p, int(rng.integers(1, 4))).copy() for p in pieces]
    for attempt in range(18):
        canvas = np.full((28, 28), -1, dtype=np.int8)
        blocked = np.zeros((28, 28), dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt:
            order = list(map(int, rng.permutation(len(pieces))))
        maxr = maxc = 0
        failed = False
        for i in order:
            p = oriented[i]
            h, w = p.shape
            rr, cc = np.nonzero(p >= 0)
            choices = []
            for r in range(29-h):
                for c in range(29-w):
                    if not np.any(blocked[r+rr, c+cc]):
                        score = max(maxr, r+h)*max(maxc, c+w) + float(rng.random())*45
                        choices.append((score, r, c))
            if not choices:
                failed = True
                break
            _, r, c = min(choices)
            canvas[r+rr, c+cc] = p[rr, cc]
            for dr, dc in NEIGHBORS+((0, 0),):
                rs, cs = r+rr+dr, c+cc+dc
                valid = (rs >= 0) & (rs < 28) & (cs >= 0) & (cs < 28)
                blocked[rs[valid], cs[valid]] = True
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not failed:
            rr, cc = np.nonzero(canvas >= 0)
            cropped = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
            h, w = cropped.shape
            extra_r = int(rng.integers(2, min(6, 30-h)+1))
            extra_c = int(rng.integers(2, min(6, 30-w)+1))
            top, left = int(rng.integers(1, extra_r)), int(rng.integers(1, extra_c))
            result = np.full((h+extra_r, w+extra_c), bg, dtype=np.int8)
            result[top:top+h, left:left+w] = np.where(cropped >= 0, cropped, bg)
            return result
    return None


def _free_cell_alternative(code, pigment_counts, side):
    """Find another answer if pigments may move independently of pieces."""
    capacities = Counter(map(int, _depths(side).flat))
    for i, j in combinations(range(len(code)), 2):
        if code[i] == code[j]:
            continue
        other = list(code)
        other[i], other[j] = other[j], other[i]
        available = Counter()
        for d, color in enumerate(other):
            available[color] += capacities[d] - int(d == 0)
        if all(available[c] >= n for c, n in pigment_counts.items()):
            return tuple(other)
    return None


def _make_pair(rng, side, count, style, bg, evidence=False, sparse=False, max_fraction=None):
    palette = [c for c in range(10) if c not in (bg, NEUTRAL, ANCHOR)]
    layers = (side+1)//2
    depths = _depths(side)
    for attempt in range(100):
        method = style if attempt < 60 else "slabs"
        labels = _partition(rng, side, count, method)
        if labels is None or min(int(np.sum(labels == i)) for i in range(count)) < 3:
            continue
        pigments = list(map(int, rng.choice(palette, int(rng.integers(2, min(layers, 5)+1)), replace=False)))
        code = list(map(int, rng.choice(pigments, layers)))
        if evidence:
            # Thick band, a return at the center, and a nonpalindromic code.
            a, b = pigments[:2]
            code = [a, a] + [b]*(layers-3) + [a]
        if len(set(code)) < 2 or code == code[::-1]:
            continue
        painting = _wrap(code, side)
        probability = float(rng.uniform(.20, min(.38, max_fraction)) if sparse else rng.uniform(.45, .65))
        visible = rng.random((side, side)) < probability
        visible[0, 0] = False
        # Every layer and every piece must have actual pigment evidence.
        for d in range(layers):
            choices = np.flatnonzero((depths == d).ravel())
            choices = choices[choices != 0]
            if not np.any(visible.ravel()[choices]):
                visible.flat[int(rng.choice(choices))] = True
        for i in range(count):
            choices = np.flatnonzero((labels == i).ravel())
            choices = choices[choices != 0]
            if not np.any(visible.ravel()[choices]):
                visible.flat[int(rng.choice(choices))] = True
        broken = np.where(visible, painting, NEUTRAL).astype(np.int8)
        broken[0, 0] = ANCHOR
        actual_fraction = float(visible.sum()/(side*side-1))
        if (sparse and actual_fraction > max_fraction) or (not sparse and actual_fraction < .38):
            continue
        pieces = [_tight(broken, np.argwhere(labels == i)) for i in range(count)]
        if any(len(_components(p >= 0)) != 1 or len(_components(p >= 0, EDGE)) > 3 for p in pieces):
            continue
        if not np.any(broken == NEUTRAL):
            continue
        counts = Counter(int(v) for v in broken.flat if v not in (NEUTRAL, ANCHOR))
        if evidence and _free_cell_alternative(code, counts, side) is None:
            continue
        answers = _assemblies(pieces, side)
        if len(answers) != 1 or answers[0][0] != tuple(code):
            continue
        if evidence and len(_assemblies(pieces, side, radial=False)) < 2:
            continue
        inp = _scatter(rng, pieces, bg)
        if inp is not None:
            return {"input": inp, "output": painting}
    raise RuntimeError("bounded construction exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # These affect distributions only. All semantic roles are input-visible.
    bg_values = list(map(int, rng.permutation([0, 1, 2, 3, 4, 6, 7, 8])))
    regime = int(rng.integers(3))
    first_side = int(rng.choice([8, 10]))
    puzzle = {"train": [], "test": []}
    seen = set()
    diagonal_seen = False
    train_fractions = []
    styles = ["slabs", "growth", "cuts", "diagonal"]
    for index in range(ntrain+ntest):
        train = index < ntrain
        if train:
            side = first_side if index == 0 else int(rng.choice([s for s in range(4, 11) if index != 1 or s != first_side]))
            count = int(rng.choice([3, 4])) if regime == 1 else int(rng.integers(3, 6))
            style = "slabs" if index == 0 else styles[index % 4]
        else:
            side = 12 if regime == 0 else int(rng.integers(7, 12))
            count = 5 if regime == 1 else int(rng.integers(3, 6))
            allowed_styles = styles if diagonal_seen else styles[:3]
            style = allowed_styles[int(rng.integers(len(allowed_styles)))]
        count = min(count, side)
        bg = bg_values[index % len(bg_values)]
        for _ in range(8):
            pair = _make_pair(rng, side, count, style, bg, evidence=train and index == 0,
                              sparse=not train and regime == 2,
                              max_fraction=min(train_fractions)-.05 if not train and regime == 2 else None)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if train else "test"].append(pair)
                if train:
                    _, _, _, visible_pieces = _read_input(pair["input"])
                    diagonal_seen |= any(len(_components(p >= 0, EDGE)) > 1 for p in visible_pieces)
                    train_fractions.append(float(np.sum((pair["input"] != bg) &
                                             (pair["input"] != NEUTRAL) & (pair["input"] != ANCHOR))/(side*side-1)))
                break
        else:
            raise RuntimeError("distinct input retries exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def _facts(pair, check_ablation=False):
    inp, bg, side, pieces = _read_input(pair["input"])
    out = _grid(pair["output"])
    _require(out.shape == (side, side), "wrong output shape")
    _require(out[0, 0] == ANCHOR and np.sum(out == ANCHOR) == 1, "anchor moved incorrectly")
    _require(not np.any(out == NEUTRAL) and not np.any(out == bg), "material/background left in answer")
    answers = _assemblies(pieces, side)
    _require(len(answers) == 1 and -1 not in answers[0][0], "ambiguous or incomplete ring code")
    code, raw = answers[0]
    depth = _depths(side)
    expected = np.asarray(code, dtype=np.int8)[depth]
    expected[0, 0] = ANCHOR
    _require(np.array_equal(out, expected), "wrong concentric completion")
    samples = (raw != NEUTRAL) & (raw != ANCHOR)
    _require(np.array_equal(raw[samples], out[samples]), "a pigment stamp changed")
    for d in range(len(code)):
        _require(bool(np.any(samples & (depth == d))), "unsampled depth")
    _require(np.sum(raw == NEUTRAL) == np.sum(inp == NEUTRAL), "material lost during assembly")
    _require(sum(int(np.sum(p >= 0)) for p in pieces) == out.size, "area not conserved")
    counts = Counter(int(v) for v in inp.flat if v not in (bg, NEUTRAL, ANCHOR))
    alternative = _free_cell_alternative(code, counts, side)
    geometry_ambiguous = len(_assemblies(pieces, side, radial=False)) > 1 if check_ablation else None
    normal = {}
    normalized_code = tuple(normal.setdefault(c, len(normal)) for c in code)
    return {"side": side, "count": len(pieces), "bg": bg, "code": code,
            "normalized_code": normalized_code, "neutral": int(np.sum(inp == NEUTRAL)),
            "pigment_fraction": float(samples.sum()/(out.size-1)),
            "repeated": any(a == b for a, b in zip(code, code[1:])),
            "returned": len(set(code)) < 1+sum(a != b for a, b in zip(code, code[1:])),
            "diagonal": sum(len(_components(p >= 0, EDGE)) > 1 for p in pieces),
            "free_cell_alternative": alternative, "geometry_ambiguous": geometry_ambiguous,
            "shape_signature": tuple(sorted((p.shape, (p >= 0).tobytes()) for p in pieces))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_facts, test_facts = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _facts(pair, check_ablation=split == "train")
            if split == "train":
                train_facts.append(f)
            else:
                test_facts.append(f)
    _require(len({f["side"] for f in train_facts}) >= 2, "demonstrate size inference")
    _require(len({f["bg"] for f in train_facts}) >= 2, "demonstrate changing background")
    _require(any(f["repeated"] and f["returned"] and f["geometry_ambiguous"]
                 and f["free_cell_alternative"] is not None for f in train_facts),
             "missing coupled-rule and ablation demonstration")
    _require(not any(f["diagonal"] for f in test_facts) or any(f["diagonal"] for f in train_facts),
             "diagonal attachments need a training demonstration")
    _require(all(f["side"] > max(t["side"] for t in train_facts) for f in test_facts)
             or all(f["count"] > max(t["count"] for t in train_facts) for f in test_facts)
             or all(f["pigment_fraction"] <= min(t["pigment_fraction"] for t in train_facts)-.049 for f in test_facts),
             "missing visible train/test shift")


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(p[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streaming measurements; no saved puzzles or execution logs."""
    signatures, shapes, geometries, codes, colors = set(), set(), set(), set(), set()
    sides, counts = set(), set()
    diagonal = repeated = returned = odd = 0
    sparse_shifts = larger_shifts = count_shifts = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        if seed == 0:
            first = puzzle
        training, testing = [], []
        for split in ("train", "test"):
            for p in puzzle[split]:
                _require(np.array_equal(solve(p["input"]), p["output"]), "solver mismatch")
                f = _facts(p)
                (training if split == "train" else testing).append(f)
                shapes.add(p["input"].shape)
                geometries.add(f["shape_signature"])
                codes.add(f["normalized_code"])
                colors.update(map(int, np.unique(p["input"])))
                sides.add(f["side"])
                counts.add(f["count"])
                diagonal += f["diagonal"] > 0
                repeated += f["repeated"]
                returned += f["returned"]
                odd += f["side"] % 2
        larger_shifts += min(f["side"] for f in testing) > max(f["side"] for f in training)
        count_shifts += min(f["count"] for f in testing) > max(f["count"] for f in training)
        sparse_shifts += min(f["pigment_fraction"] for f in training) > max(f["pigment_fraction"] for f in testing)
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(geometries) >= 100, "insufficient structural variation")
    _require(len(codes) >= 18 and sides == set(range(4, 13)) and counts == {3, 4, 5}, "missing code/scale/count variation")
    _require(colors == set(range(10)) and diagonal >= 5 and repeated >= 42 and returned >= 42 and odd >= 20, "missing family features")
    _require(larger_shifts >= 5 and count_shifts >= 5 and sparse_shifts >= 5, "missing train/test regimes")
    _require(_puzzle_digest(first) == _puzzle_digest(generate_puzzle(0)), "nondeterminism")
    # Metamorphic checks: scatter orientation and pigment names carry no code.
    pair = first["test"][0]
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(pair["input"], k)), pair["output"]), "scatter rotation changed answer")
    mapping = np.arange(10, dtype=np.int8)
    moving = [c for c in range(10) if c not in (NEUTRAL, ANCHOR)]
    mapping[moving] = moving[::-1]
    _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "palette permutation changed rule")
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][1, 1] = NEUTRAL
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt painting accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "piece_geometries": len(geometries), "normalized_codes": len(codes),
            "diagonal_pairs": diagonal, "repeated_pairs": repeated, "returned_pairs": returned,
            "odd_pairs": odd, "larger_test_puzzles": larger_shifts,
            "more_piece_test_puzzles": count_shifts, "sparser_test_puzzles": sparse_shifts}
