"""Folding mold: reflected negative space becomes a rigid packing surface.

The hinge color encodes the material of the new mold.  No learned palette,
seed, example table, or generation metadata is needed by solve().
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int64, copy=True)


def _components(a, background):
    seen = set()
    result = []
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or a[r, c] == background:
                continue
            color = int(a[r, c])
            seen.add((r, c))
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen
                            and a[yy, xx] == color):
                        seen.add((yy, xx))
                        stack.append((yy, xx))
            result.append((color, tuple(sorted(cells))))
    return result


def _order(pieces):
    """Near-hinge pieces precede all pieces behind them on a shared row."""
    spans = []
    for _, cells in pieces:
        rows = {}
        for r, c in cells:
            lo, hi = rows.get(r, (c, c))
            rows[r] = min(lo, c), max(hi, c)
        spans.append(rows)
    before = [set() for _ in pieces]
    for i in range(len(pieces)):
        for j in range(i):
            common = spans[i].keys() & spans[j].keys()
            if not common:
                continue
            i_first = all(spans[i][r][1] < spans[j][r][0] for r in common)
            j_first = all(spans[j][r][1] < spans[i][r][0] for r in common)
            _require(i_first != j_first, "interleaved pieces")
            before[j if i_first else i].add(i if i_first else j)
    pending, answer = set(range(len(pieces))), []
    while pending:
        ready = [i for i in pending if not (before[i] & pending)]
        _require(bool(ready), "cyclic piece precedence")
        # Incomparable pieces have disjoint row supports; this tie has no effect.
        i = min(ready, key=lambda j: (min(c for _, c in pieces[j][1]), j))
        answer.append(i)
        pending.remove(i)
    return answer


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    _require(h >= 5 and w >= 8, "canvas too small for a hinge")
    bg = int(a[0, 0])
    hinges = [c for c in range(w) if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    _require(len(hinges) == 1, "expected one full-height hinge")
    hinge = hinges[0]
    wall = int(a[0, hinge])
    _require(hinge > 0 and hinge < w - 1, "hinge must be internal")
    _require(hinge != w - hinge - 1, "source side must be uniquely narrower")
    flipped = hinge > w - hinge - 1
    if flipped:
        a = np.fliplr(a).copy()
        hinge = w - hinge - 1
    k = hinge
    _require(k >= 2 and w - k - 1 > k, "invalid stencil/workspace widths")
    _require(np.all(a[[0, -1], :k] == bg) and np.all(a[[0, -1], k + 1:] == bg),
             "top and bottom margins must be empty except for hinge")
    source = a[:, :k]
    other = set(map(int, np.unique(source))) - {bg}
    _require(len(other) == 1 and wall not in other, "source must be a binary stencil")
    stencil = other.pop()
    _require(bool(np.any(source[:, 0] == stencil)), "stencil must touch outer edge")
    chamber = a[:, k + 1:]
    _require(not np.any(chamber == wall), "hinge color is reserved for fixed material")
    pieces = _components(chamber, bg)
    _require(bool(pieces), "no loose pieces")
    order = _order(pieces)
    mold = np.zeros(chamber.shape, dtype=bool)
    mold[:, :k] = np.fliplr(source == bg)
    _require(not np.any(mold & (chamber != bg)), "a piece overlaps the future mold")
    return a, bg, wall, stencil, k, flipped, chamber, mold, pieces, order


def _pack(chamber, bg, wall, mold, pieces, order, independent=False):
    """Translate one column at a time, so no piece can cross a solid cell."""
    frozen_mold = set(map(tuple, np.argwhere(mold)))
    frozen = set(frozen_mold)
    out = np.full(chamber.shape, bg, dtype=int)
    out[mold] = wall
    shifts = [0] * len(pieces)
    for i in order:
        color, cells = pieces[i]
        obstacles = frozen_mold if independent else frozen
        shift = 0
        while all(c - shift - 1 >= 0 and (r, c - shift - 1) not in obstacles
                  for r, c in cells):
            shift += 1
        shifts[i] = shift
        for r, c in cells:
            out[r, c - shift] = color
            frozen.add((r, c - shift))
    return out, tuple(shifts)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, wall, _, k, flipped, chamber, mold, pieces, order = _parse(input_grid)
    packed, _ = _pack(chamber, bg, wall, mold, pieces, order)
    out = a.copy()
    out[:, k + 1:] = packed
    return np.fliplr(out).copy() if flipped else out


def _rowwise(chamber, bg, wall, mold):
    """Counterfactual: every colored pixel is free to move independently."""
    pieces = [(int(chamber[r, c]), ((int(r), int(c)),))
              for r, c in np.argwhere(chamber != bg)]
    return _pack(chamber, bg, wall, mold, pieces, _order(pieces))[0]


def _features(value):
    a, bg, wall, _, k, flipped, chamber, mold, pieces, order = _parse(value)
    packed, shifts = _pack(chamber, bg, wall, mold, pieces, order)
    blank = np.zeros(chamber.shape, dtype=bool)
    _, no_mold = _pack(chamber, bg, wall, blank, pieces, order)
    _, independent = _pack(chamber, bg, wall, mold, pieces, order, independent=True)
    unreflected = blank.copy()
    unreflected[:, :k] = a[:, :k] == bg
    _, no_mirror = _pack(chamber, bg, wall, unreflected, pieces, order)
    positive = blank.copy()
    positive[:, :k] = np.fliplr(a[:, :k] != bg)
    _, no_complement = _pack(chamber, bg, wall, positive, pieces, order)
    # Replace every stencil row by a run of the same population at the outer edge.
    counted = blank.copy()
    for r in range(len(a)):
        n = int(np.count_nonzero(a[r, :k] != bg))
        counted[r, :k - n] = True
    _, count_only = _pack(chamber, bg, wall, counted, pieces, order)
    return {
        "source_width": k, "side": "right" if flipped else "left",
        "pieces": len(pieces), "moved": sum(s > 0 for s in shifts),
        "stationary": sum(s == 0 for s in shifts), "max_shift": max(shifts),
        "enters_recess": any(c - shifts[i] < k for i, (_, cells) in enumerate(pieces) for _, c in cells),
        "mold_changes_motion": shifts != no_mold,
        "mirror_changes_motion": shifts != no_mirror,
        "complement_changes_motion": shifts != no_complement,
        "count_shortcut_fails": shifts != count_only,
        "piece_collision": shifts != independent,
        "rigidity_matters": not np.array_equal(packed, _rowwise(chamber, bg, wall, mold)),
        "shapes": tuple(tuple((r - min(y for y, _ in cells), c - min(x for _, x in cells))
                              for r, c in cells) for _, cells in pieces),
        "shifts": shifts,
    }


def _stencil(rng, h, k, style, demonstration):
    mask = np.zeros((h, k), dtype=bool)
    if style in ("stepped", "cutout"):
        lengths = rng.integers(0, k + 1, size=h - 2)
        if rng.random() < 0.5:
            for r in range(1, len(lengths)):
                lengths[r] = np.clip(lengths[r - 1] + int(rng.integers(-2, 3)), 0, k)
        for r, length in enumerate(lengths, 1):
            mask[r, :int(length)] = True
        if style == "cutout":
            # Islands beyond gaps make total population insufficient to predict reach.
            mask[1:-1, 1:] ^= rng.random((h - 2, k - 1)) < 0.28
    elif style == "patches":
        for _ in range(int(rng.integers(2, 7))):
            r = int(rng.integers(1, h - 1))
            c = int(rng.integers(k))
            mask[r:min(h - 1, r + int(rng.integers(1, 5))),
                 c:min(k, c + int(rng.integers(1, k + 1)))] = True
    else:
        mask[1:-1] = rng.random((h - 2, k)) < float(rng.uniform(0.25, 0.85))
    # Keep both a deep pocket and a partial-depth pocket in the demonstration.
    if demonstration:
        r = int(rng.integers(1, h - 3))
        mask[r:r + 2, :] = True
        candidates = [y for y in range(1, h - 1) if y not in (r, r + 1)]
        y = int(rng.choice(candidates))
        mask[y] = False
        mask[y, :int(rng.integers(1, k))] = True
        if style == "cutout":
            y = int(rng.choice(candidates))
            mask[y] = True
            mask[y, int(rng.integers(1, k - 1))] = False
    if not np.any(mask[:, 0]):
        mask[int(rng.integers(1, h - 1)), 0] = True
    return mask


def _normalize(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape(rng, large=False, complex_shape=False):
    kind = int(rng.choice(5, p=(0.20, 0.23, 0.12, 0.20, 0.25)))
    if complex_shape:
        kind = int(rng.choice((3, 4)))
    if kind == 0:
        mask = np.ones((1, 1), dtype=bool)
    elif kind == 1:
        mask = np.ones((1, int(rng.integers(2, 7 if large else 5))), dtype=bool)
    elif kind == 2:
        mask = np.ones((int(rng.integers(2, 4)), int(rng.integers(2, 4))), dtype=bool)
    elif kind == 3:
        patterns = (("11", "10"), ("111", "010"), ("101", "111"),
                    ("110", "011"), ("010", "111", "010"),
                    ("111", "100", "100"), ("1111", "0100"))
        lines = patterns[int(rng.integers(len(patterns)))]
        mask = np.array([[c == "1" for c in row] for row in lines])
    else:
        cells = {(0, 0)}
        n = int(rng.integers(4, 13 if large else 9))
        for _ in range(n - 1):
            candidates = sorted({(r + dr, c + dc) for r, c in cells
                                 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
            cells.add(candidates[int(rng.integers(len(candidates)))])
        cells = _normalize(cells)
        mask = np.zeros((max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.integers(2):
        mask = np.fliplr(mask)
    return _normalize(map(tuple, np.argwhere(mask)))


def _can_place(cells, color, placed):
    own = set(cells)
    for old_color, old_cells in placed:
        other = set(old_cells)
        if own & other:
            return False
        if old_color == color and any((r + dr, c + dc) in other for r, c in own
                                      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            return False
    try:
        _order(placed + [(color, cells)])
    except ValueError:
        return False
    return True


def _make_pair(rng, palette, side, style, demonstration, test):
    bg, wall, ink, colors = palette
    for _ in range(600):
        h = int(rng.integers(9, 24 if test else 18))
        k = int(rng.integers(3, 8 if test else 7))
        length = int(rng.integers(k + 7, 30 - k))
        mask = _stencil(rng, h, k, style, demonstration)
        n = int(rng.integers(6, 10)) if test else int(rng.integers(3 if demonstration else 2, 6))
        shapes = [_shape(rng, large=test, complex_shape=demonstration and i == 0) for i in range(n)]
        piece_colors = [int(rng.choice(colors)) for _ in range(n)]
        placed = []
        for i in sorted(range(n), key=lambda j: -len(shapes[j])):
            shape = shapes[i]
            sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
            if sh > h - 2 or sw > length - k:
                break
            for _ in range(100):
                top = int(rng.integers(1, h - sh))
                left = int(rng.integers(k, length - sw + 1))
                cells = tuple((top + r, left + c) for r, c in shape)
                if _can_place(cells, piece_colors[i], placed):
                    placed.append((piece_colors[i], cells))
                    break
            else:
                break
        if len(placed) != n:
            continue
        a = np.full((h, k + 1 + length), bg, dtype=int)
        a[:, :k][mask] = ink
        a[:, k] = wall
        for color, cells in placed:
            for r, c in cells:
                a[r, k + 1 + c] = color
        if side == "right":
            a = np.fliplr(a).copy()
        f = _features(a)
        if not (f["moved"] and f["mold_changes_motion"]):
            continue
        if demonstration and not (f["enters_recess"] and f["rigidity_matters"]
                                  and f["piece_collision"] and f["mirror_changes_motion"]
                                  and f["complement_changes_motion"]):
            continue
        if demonstration and style == "cutout" and not f["count_shortcut_fails"]:
            continue
        return {"input": a, "output": solve(a)}
    raise RuntimeError("bounded sampler could not construct the requested demonstration")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _key(a):
    a = np.asarray(a, dtype=np.int64)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.choice(10, 3, replace=False)))
    mobile_palette = [c for c in range(10) if c not in roles[:2]]
    colors = tuple(map(int, rng.choice(mobile_palette, int(rng.integers(2, 6)), replace=False)))
    palette = (*roles, colors)
    styles = ["stepped", "cutout"]
    rng.shuffle(styles)
    sides = ["left", "right"]
    rng.shuffle(sides)
    p, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            diagnostic = split == "train" and i < 2
            style = styles[i] if diagnostic else str(rng.choice(("stepped", "cutout", "patches", "islands")))
            side = sides[i] if diagnostic else str(rng.choice(("left", "right")))
            for _ in range(20):
                pair = _make_pair(rng, palette, side, style, diagnostic, split == "test")
                signature = _key(pair["input"])
                if signature not in used:
                    p[split].append(pair)
                    used.add(signature)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(p)
    return p


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, roles, train_features = set(), set(), []
    train_counts, test_counts = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _key(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a, bg, wall, ink, k, flip, chamber, mold, pieces, order = _parse(inp)
            canonical_out = np.fliplr(out) if flip else out
            _require(out.shape == inp.shape, "output shape changed")
            roles.add((bg, wall, ink))
            _require(np.array_equal(canonical_out[:, :k + 1], a[:, :k + 1]), "stencil or hinge changed")
            actual = canonical_out[:, k + 1:]
            expected, shifts = _pack(chamber, bg, wall, mold, pieces, order)
            _require(np.array_equal(actual, expected), "incorrect packing")
            _require(np.array_equal(actual == wall, mold), "mold must be the complete reflected negative")
            for r in range(len(chamber)):
                old = Counter(map(int, chamber[r][chamber[r] != bg]))
                new = Counter(map(int, actual[r][(actual[r] != bg) & (actual[r] != wall)]))
                _require(old == new, "mobile row colors/counts changed")
            occupied = set(map(tuple, np.argwhere(actual != bg)))
            for i, (color, cells) in enumerate(pieces):
                moved = {(r, c - shifts[i]) for r, c in cells}
                _require(all(actual[r, c] == color for r, c in moved), "piece lost rigidity or color")
                _require(any(c == 0 or (r, c - 1) in occupied - moved for r, c in moved), "unsettled piece")
            f = _features(inp)
            _require(f["moved"] > 0 and f["mold_changes_motion"], "both inherited mechanisms must affect pair")
            (train_counts if split == "train" else test_counts).append(len(pieces))
            if split == "train":
                train_features.append(f)
    _require(len(roles) == 1, "puzzle palette roles changed")
    _require({f["side"] for f in train_features} == {"left", "right"}, "training must demonstrate both directions")
    _require(any(f["rigidity_matters"] and f["piece_collision"] and f["enters_recess"]
                 and f["mirror_changes_motion"] and f["complement_changes_motion"]
                 for f in train_features), "missing mechanical demonstration")
    _require(any(f["count_shortcut_fails"] for f in train_features), "training does not distinguish counts from geometry")
    _require(min(test_counts) > max(train_counts), "missing held-out crowding shift")


def _signature(puzzle):
    return tuple(_key(pair[role]) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def self_test():
    """Bounded semantic, metamorphic, count-contract, and 42-seed checks."""
    # Hand-worked mold: source rows 110 and 011 have equal populations but
    # different entrances. The vertical 6 piece stops as a unit; 7 queues behind it.
    given = ("0002000000000", "1112000000030", "1002000000440", "1012000000400",
             "1102000006000", "0112000006070", "0002000000000")
    answer = ("0002222000000", "1112300000000", "1002224400000", "1012024000000",
              "1102200600000", "0112002670000", "0002222000000")
    fixture = np.array([[int(c) for c in row] for row in given])
    expected = np.array([[int(c) for c in row] for row in answer])
    _require(np.array_equal(solve(fixture), expected), "hand-worked docking fixture failed")
    dims, shapes, stencils, palettes, signatures = set(), set(), set(), set(), set()
    counts, source_widths, travel = set(), set(), set()
    totals = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                a, bg, wall, ink, k, flip, chamber, mold, pieces, order = _parse(inp)
                f = _features(inp)
                dims.add(inp.shape)
                shapes.update(f["shapes"])
                stencils.add((a.shape[0], k, (a[:, :k] != bg).tobytes()))
                palettes.add((bg, wall, ink))
                counts.add(len(pieces))
                source_widths.add(k)
                travel.update(f["shifts"])
                for name in ("stationary", "enters_recess", "piece_collision", "rigidity_matters", "count_shortcut_fails"):
                    totals[name] += int(f[name])
                _require(np.array_equal(solve(inp), out), "solve mismatch")
                # Horizontal equivariance checks orientation inference and mold placement.
                _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "reflection equivariance failed")
                # Color permutations must commute with the rule, including background.
                perm = np.random.default_rng(seed + 500).permutation(10)
                _require(np.array_equal(solve(perm[inp]), perm[out]), "palette equivariance failed")
                # Cell-by-cell check of the new material, separate from packing.
                actual = (np.fliplr(out) if flip else out)[:, k + 1:]
                for r in range(len(a)):
                    for c in range(chamber.shape[1]):
                        is_solid = c < k and a[r, k - 1 - c] == bg
                        _require((actual[r, c] == wall) == is_solid, "incorrect negative cell")
    _require(len(signatures) == 42 and len(dims) >= 50, "insufficient canvas variation")
    _require(len(shapes) >= 50 and len(stencils) >= 180, "insufficient geometric variation")
    _require(len(palettes) >= 30 and len(counts) >= 7 and len(source_widths) >= 5, "insufficient role/count variation")
    _require(0 in travel and 1 in travel and max(travel) >= 10, "insufficient travel variation")
    _require(all(totals[k] > 0 for k in totals), "missing interaction mode")
    _require(_signature(generate_puzzle(1337)) == _signature(generate_puzzle(1337)), "nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + nt * 10 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "wrong explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(dims),
            "piece_shapes": len(shapes), "stencil_masks": len(stencils),
            "role_palettes": len(palettes), "piece_counts": sorted(counts),
            "stencil_widths": sorted(source_widths), "travel_range": [min(travel), max(travel)],
            "interaction_totals": dict(totals)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
