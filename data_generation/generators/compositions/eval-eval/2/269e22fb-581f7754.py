"""Dock incomplete patterned tiles using registration pins recovered from exemplars.

Only the returned grids carry solving information. No parent generator is imported.
"""
from __future__ import annotations

import copy
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer colors required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _d4(a, k):
    # Four rotations, then those four rotations after a left-right reflection.
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _swap(a, body, swap):
    if not swap:
        return a.copy()
    return np.where(a == body[0], body[1], np.where(a == body[1], body[0], a)).astype(np.int8)


def _components(a, bg):
    left = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while left:
        start = min(left)
        left.remove(start)
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _bbox(points):
    return (min(r for r, _ in points), min(c for _, c in points),
            max(r for r, _ in points) + 1, max(c for _, c in points) + 1)


def _decode(a):
    h, w = a.shape
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(counts.max() > a.size / 2, "background must occupy over half the input")
    components = _components(a, bg)
    anchors, other = {}, []
    fixed = np.full_like(a, bg)
    for cells in components:
        if len(cells) == 1:
            r, c = next(iter(cells))
            _require(r in (0, h - 1) or c in (0, w - 1), "interior singleton")
            _require(not (r in (0, h - 1) and c in (0, w - 1)), "corner anchor")
            anchors.setdefault(int(a[r, c]), []).append((r, c))
            fixed[r, c] = a[r, c]
        else:
            other.append(cells)
    _require(1 <= len(anchors) <= 2, "one or two guide identities required")
    guides = {}
    for color, points in anchors.items():
        horizontal = len({r for r, _ in points}) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len({c for _, c in points}) == 1 and all(r in (0, h - 1) for r, _ in points)
        _require(horizontal != vertical and len(points) <= 2, "inconsistent anchors")
        axis = 0 if horizontal else 1
        guides[color] = (axis, points[0][axis])
    references, fragments = [], []
    body, frame = None, None
    for cells in other:
        r, c, r1, c1 = _bbox(cells)
        tile = a[r:r1, c:c1]
        _require(len(cells) == tile.size, "components must be solid rectangles")
        anchored = set(map(int, np.unique(tile))) & set(anchors)
        if anchored:
            _require(len(anchored) == 1 and min(tile.shape) >= 6, "invalid exemplar")
            marker = next(iter(anchored))
            border = np.concatenate((tile[0], tile[-1], tile[1:-1, 0], tile[1:-1, -1]))
            _require(len(np.unique(border)) == 1, "exemplar must have a uniform frame")
            f = int(border[0])
            inside = tile[1:-1, 1:-1].copy()
            colors = set(map(int, np.unique(inside)))
            _require(f not in colors and f not in anchors and len(colors) == 3,
                     "frame and interior roles must be distinct")
            _require(np.sum(inside == marker) == 1, "exemplar needs exactly one pin")
            b = tuple(sorted(colors - {marker}))
            _require(not (set(b) & set(anchors)), "body colors cannot be guide identities")
            if body is None:
                body, frame = b, f
            _require(body == b and frame == f, "exemplars must share body and frame colors")
            references.append({"tile": inside, "marker": marker, "box": (r, c, r1, c1)})
            fixed[r:r1, c:c1] = tile
        else:
            _require(min(tile.shape) >= 2 and len(np.unique(tile)) == 2, "invalid binary fragment")
            fragments.append({"patch": tile.copy(), "origin": (r, c)})
    _require(len(references) == len(anchors) and {x["marker"] for x in references} == set(anchors),
             "each guide needs exactly one framed exemplar")
    _require(1 <= len(fragments) <= 4, "one to four fragments required")
    _require(all(tuple(map(int, np.unique(x["patch"]))) == body for x in fragments), "fragment palette")
    return bg, body, guides, references, fragments, fixed


def _atlas(references, body, shapes=None):
    """Exact matches, including location, identity, D4 orientation and role exchange.

    A repeated key maps to None; ambiguity is rejected rather than broken by order.
    With shapes supplied, enumerate only the query shapes, keeping solving bounded.
    """
    atlas = {}
    for j, ref in enumerate(references):
        for k in range(8):
            for swap in (0, 1):
                full = _swap(_d4(ref["tile"], k), body, swap)
                H, W = full.shape
                pr, pc = map(int, np.argwhere(full == ref["marker"])[0])
                sizes = shapes if shapes is not None else [(h, w) for h in range(2, H + 1) for w in range(2, W + 1)]
                for h, w in sizes:
                    for r in range(H - h + 1):
                        for c in range(W - w + 1):
                            if r <= pr < r + h and c <= pc < c + w:
                                continue
                            patch = full[r:r + h, c:c + w]
                            if patch.min() == patch.max():
                                continue
                            key = _key(patch)
                            record = (j, k, swap, r, c)
                            atlas[key] = None if key in atlas else record
    return atlas


def _infer(a):
    bg, body, guides, references, fragments, fixed = _decode(a)
    atlas = _atlas(references, body, {x["patch"].shape for x in fragments})
    moves = []
    used = set()
    for fragment in fragments:
        record = atlas.get(_key(fragment["patch"]))
        _require(record is not None, "fragment has no unique exact match")
        j, k, swap, r, c = record
        ref = references[j]
        full = _swap(_d4(ref["tile"], k), body, swap)
        pin = tuple(map(int, np.argwhere(full == ref["marker"])[0]))
        origin = tuple(fragment["origin"][d] - (r, c)[d] for d in (0, 1))
        virtual_pin = tuple(origin[d] + pin[d] for d in (0, 1))
        axis, guide = guides[ref["marker"]]
        delta = [0, 0]
        delta[axis] = guide - virtual_pin[axis]
        target_origin = tuple(origin[d] + delta[d] for d in (0, 1))
        tangent = 1 - axis
        _require(0 <= target_origin[tangent] and target_origin[tangent] + full.shape[tangent] <= a.shape[tangent],
                 "completion must fit along its guide")
        target = {}
        clipped = 0
        for rr in range(full.shape[0]):
            for cc in range(full.shape[1]):
                p = target_origin[0] + rr, target_origin[1] + cc
                if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]:
                    target[p] = int(full[rr, cc])
                else:
                    clipped += 1
        moves.append({**fragment, "ref": j, "orientation": k, "swap": swap, "offset": (r, c),
                      "full": full, "pin": pin, "virtual_pin": virtual_pin, "axis": axis,
                      "delta": tuple(delta), "target_origin": target_origin, "target": target,
                      "clipped": clipped, "fraction": fragment["patch"].size / full.size})
        used.add(j)
    _require(used == set(range(len(references))), "unused exemplar")
    return bg, body, guides, references, fixed, moves


def _render(fixed, bg, moves, complete=True, move=True):
    result = fixed.copy()
    occupied = result != bg
    for m in moves:
        if complete and move:
            target = m["target"]
        elif complete:
            origin = tuple(m["origin"][d] - m["offset"][d] for d in (0, 1))
            target = {(origin[0] + r, origin[1] + c): int(m["full"][r, c])
                      for r in range(m["full"].shape[0]) for c in range(m["full"].shape[1])}
        else:
            origin = tuple(m["origin"][d] + m["delta"][d] for d in (0, 1))
            target = {(origin[0] + r, origin[1] + c): int(m["patch"][r, c])
                      for r in range(m["patch"].shape[0]) for c in range(m["patch"].shape[1])}
        for p, color in target.items():
            if not (0 <= p[0] < result.shape[0] and 0 <= p[1] < result.shape[1]):
                continue
            _require(not occupied[p], "destination overlaps fixed material or another tile")
            occupied[p] = True
            result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    bg, _, _, _, fixed, moves = _infer(a)
    return _render(fixed, bg, moves)


def _pattern(rng, h, w):
    """Small structured designs: loops, bars, bends, combs and staircases combine."""
    a = np.zeros((h, w), np.int8)
    for _ in range(int(rng.integers(3, 7))):
        kind = int(rng.integers(5))
        r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        if kind == 0:
            a[r0, c0:c1 + 1] = 1
            if rng.random() < .5:
                a[r0:r1 + 1, c1] = 1
        elif kind == 1:
            a[r0:r1 + 1, c0] = 1
            a[r1, c0:c1 + 1] = 1
        elif kind == 2:
            a[r0:r1 + 1, (c0, c1)] = 1
            a[(r0, r1), c0:c1 + 1] = 1
        elif kind == 3:
            a[r0:r1 + 1, c0] = 1
            for r in range(r0, r1 + 1, int(rng.integers(2, 4))):
                a[r, c0:c1 + 1] = 1
        else:
            for r in range(r0, r1 + 1):
                c = c0 + (r - r0) * (c1 - c0) // max(1, r1 - r0)
                a[r, c] = 1
                if r < r1:
                    nc = c0 + (r + 1 - r0) * (c1 - c0) // max(1, r1 - r0)
                    a[r, c:nc + 1] = 1
    return _d4(a, int(rng.integers(8)))


def _templates(rng, n, palette):
    bg, frame, b0, b1 = palette[:4]
    body = tuple(sorted((b0, b1)))
    for _ in range(100):
        refs = []
        for j in range(n):
            h, w = map(int, rng.integers(5, 9, size=2))
            if h == w == 5:
                w = 6
            bits = _pattern(rng, h, w)
            h, w = bits.shape
            tile = np.where(bits, b1, b0).astype(np.int8)
            pr, pc = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            tile[pr, pc] = palette[4 + j]
            refs.append({"tile": tile, "marker": palette[4 + j]})
        atlas = _atlas(refs, body)
        banks = [[(r, c, key[0][0], key[0][1]) for key, value in atlas.items()
                  if value is not None and value[:3] == (j, 0, 0)
                  for r, c in [value[3:]]] for j in range(n)]
        if all(any(.45 <= h * w / refs[j]["tile"].size <= .85 for _, _, h, w in bank)
               and any(.18 <= h * w / refs[j]["tile"].size <= .40 for _, _, h, w in bank)
               for j, bank in enumerate(banks)):
            return refs, banks, body
    raise RuntimeError("could not construct identifiable structured exemplars")


def _inflate(mask):
    b = mask.copy()
    b[1:] |= mask[:-1]
    b[:-1] |= mask[1:]
    b[:, 1:] |= mask[:, :-1]
    b[:, :-1] |= mask[:, 1:]
    return b


def _clear(mask, r, c, h, w):
    return not bool(np.any(mask[r:r + h, c:c + w]))


def _crop_copy(ref, box, orientation, swap, body):
    r, c, h, w = box
    mask = np.zeros(ref["tile"].shape, np.int8)
    mask[r:r + h, c:c + w] = 1
    mask = _d4(mask, orientation)
    rr, cc = np.nonzero(mask)
    r, c = int(rr.min()), int(cc.min())
    h, w = int(rr.max() - r + 1), int(cc.max() - c + 1)
    full = _swap(_d4(ref["tile"], orientation), body, swap)
    pin = tuple(map(int, np.argwhere(full == ref["marker"])[0]))
    return {"full": full, "patch": full[r:r + h, c:c + w].copy(), "offset": (r, c), "pin": pin}


def _place_fragment(rng, inp, out, bg, copy_tile, axis, guide, moving):
    H, W = inp.shape
    full, patch = copy_tile["full"], copy_tile["patch"]
    off, pin = copy_tile["offset"], copy_tile["pin"]
    tangent = 1 - axis
    blocked = _inflate(inp != bg)
    out_mask = out != bg
    options = []
    for t in range(inp.shape[tangent] - full.shape[tangent] + 1):
        dest = [0, 0]
        dest[tangent], dest[axis] = t, guide - pin[axis]
        r0, c0 = max(0, dest[0]), max(0, dest[1])
        r1, c1 = min(H, dest[0] + full.shape[0]), min(W, dest[1] + full.shape[1])
        if not _clear(out_mask, r0, c0, r1 - r0, c1 - c0):
            continue
        for s in range(inp.shape[axis] - patch.shape[axis] + 1):
            source = [0, 0]
            source[axis], source[tangent] = s, t + off[tangent]
            delta = guide - (s - off[axis] + pin[axis])
            if moving and delta == 0:
                continue
            if not moving and delta != 0:
                continue
            if _clear(blocked, source[0], source[1], *patch.shape):
                options.append((source, dest))
    if not options:
        return False
    source, dest = options[int(rng.integers(len(options)))]
    inp[source[0]:source[0] + patch.shape[0], source[1]:source[1] + patch.shape[1]] = patch
    r0, c0 = max(0, dest[0]), max(0, dest[1])
    r1, c1 = min(H, dest[0] + full.shape[0]), min(W, dest[1] + full.shape[1])
    out[r0:r1, c0:c1] = full[r0 - dest[0]:r1 - dest[0], c0 - dest[1]:c1 - dest[1]]
    return True


def _scene(rng, palette, kind, test):
    # The two witness kinds can be shuffled; their layouts and palettes are fresh.
    n = 1 if kind == 0 else 2 if kind == 1 else int(rng.integers(1, 3))
    refs, banks, body = _templates(rng, n, palette)
    requests = []
    for j, ref in enumerate(refs):
        count = 2 if kind == 0 else 1 if kind == 1 else int(rng.integers(1, 3))
        shared_box = None
        for i in range(count):
            candidates = [b for b in banks[j] if (.18 if test else .45) <= b[2] * b[3] / ref["tile"].size <= (.40 if test else .85)]
            box = shared_box if kind == 0 and shared_box is not None else candidates[int(rng.integers(len(candidates)))]
            shared_box = box
            orientation = i if kind == 0 else 4 if kind == 1 and j == 0 else int(rng.integers(8))
            swap = i if kind == 0 else int(rng.integers(2))
            requests.append({"j": j, **_crop_copy(ref, box, orientation, swap, body)})
    # Single-reference scenes also include short-wide and tall-narrow canvases.
    H, W = map(int, rng.integers(19 if n == 2 else 16, 31, size=2))
    if n == 1 and rng.random() < .22:
        H, W = int(rng.integers(14, 18)), int(rng.integers(25, 31))
        if rng.random() < .5:
            H, W = W, H
    bg, frame = palette[:2]
    inp = np.full((H, W), bg, np.int8)
    # Exemplars are scattered, not arranged in a panel or catalog row.
    for ref in refs:
        tile = np.pad(ref["tile"], 1, constant_values=frame)
        options = [(r, c) for r in range(H - tile.shape[0] + 1) for c in range(W - tile.shape[1] + 1)
                   if _clear(_inflate(inp != bg), r, c, *tile.shape)]
        if not options:
            return None
        r, c = options[int(rng.integers(len(options)))]
        inp[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
    axes = list(map(int, rng.permutation(2))) if kind == 1 else list(map(int, rng.integers(2, size=n)))
    systems = []
    for j, ref in enumerate(refs):
        axis = axes[j]
        copies = [x for x in requests if x["j"] == j]
        clip = (kind == 1 and j == 0) or (kind != 0 and rng.random() < .30)
        legal = []
        for guide in range(1, inp.shape[axis] - 1):
            outside = [guide - x["pin"][axis] < 0 or guide - x["pin"][axis] + x["full"].shape[axis] > inp.shape[axis]
                       for x in copies]
            if (clip and not outside[0]) or (not clip and any(outside)):
                continue
            legal.append(guide)
        rng.shuffle(legal)
        ends = (0, 1) if rng.random() < .35 else (int(rng.integers(2)),)
        for guide in legal:
            points = [(guide, 0 if end == 0 else W - 1) if axis == 0 else
                      (0 if end == 0 else H - 1, guide) for end in ends]
            if any(_inflate(inp != bg)[p] for p in points):
                continue
            for p in points:
                inp[p] = ref["marker"]
            systems.append((axis, guide))
            break
        else:
            return None
    out = inp.copy()
    for x in requests:
        axis, guide = systems[x["j"]]
        # Aligned virtual pins are occasional zero-displacement completions.
        moving = not (kind == 2 and rng.random() < .12)
        if not _place_fragment(rng, inp, out, bg, x, axis, guide, moving):
            return None
    if np.mean(inp == bg) <= .5:
        return None
    try:
        predicted = solve(inp)
    except ValueError:
        return None
    _require(np.array_equal(predicted, out), "construction and input-derived docking disagree")
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .35)
    kinds = [0, 1] + [2] * (nt - 2)
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plans in (("train", kinds), ("test", [2] * nq)):
        for kind in plans:
            for _ in range(120):
                colors = palette if stable else tuple(map(int, rng.permutation(10)))
                pair = _scene(rng, colors, kind, split == "test")
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape or duplicate input")
            seen.add(_key(inp))
            bg, body, guides, refs, fixed, moves = _infer(inp)
            _require(np.array_equal(_render(fixed, bg, moves), out), "incorrect docking output")
            _require(np.array_equal(out[fixed != bg], fixed[fixed != bg]), "fixed exemplars/anchors changed")
            expected_cells = int(np.sum(fixed != bg)) + sum(len(m["target"]) for m in moves)
            _require(np.sum(out != bg) == expected_cells, "lost or extra completion cells")
            _require(not np.array_equal(inp, out), "unchanged pair")
            for m in moves:
                axis = m["axis"]
                _require(m["delta"][1 - axis] == 0, "along-guide displacement")
                _require(m["target_origin"][axis] + m["pin"][axis] == guides[refs[m["ref"]]["marker"]][1], "pin misses guide")
                _require(m["patch"].size < m["full"].size, "fragment must be proper")
                _require((.45 <= m["fraction"] <= .85) if split == "train" else (.18 <= m["fraction"] <= .40), "crop exposure shift")
                # Every in-bounds crop cell survives unchanged at its translated coordinate.
                for r in range(m["patch"].shape[0]):
                    for c in range(m["patch"].shape[1]):
                        p = (m["origin"][0] + r + m["delta"][0], m["origin"][1] + c + m["delta"][1])
                        if 0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1]:
                            _require(out[p] == m["patch"][r, c], "crop recolored or distorted")
            if split == "train":
                training.append((guides, moves))
    all_moves = [m for _, moves in training for m in moves]
    _require(any(m["delta"] != (0, 0) and not m["clipped"] for m in all_moves), "no complete moving demonstration")
    _require(any(m["delta"] != (0, 0) and not m["clipped"]
                 and not 0 <= m["pin"][m["axis"]] - m["offset"][m["axis"]] < m["patch"].shape[m["axis"]]
                 for m in all_moves), "no invisible perpendicular registration witness")
    _require(any(m["swap"] for m in all_moves), "role exchange not demonstrated")
    _require(any(m["orientation"] in (1, 3) for m in all_moves), "quarter turn not demonstrated")
    _require(any(m["orientation"] >= 4 for m in all_moves), "reflection not demonstrated")
    _require(any(m["clipped"] for m in all_moves), "clipping not demonstrated")
    _require(any({v[0] for v in guides.values()} == {0, 1} for guides, _ in training), "mixed guides not demonstrated")


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed checks, geometry/color equivariance and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "puzzle diversity")
    shapes, ref_classes, crop_shapes, offsets, orientations, backgrounds = set(), set(), set(), set(), set(), set()
    systems, clips, alignment, signs = set(), set(), 0, set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                bg, body, guides, refs, fixed, moves = _infer(a)
                shapes.add(a.shape)
                backgrounds.add(bg)
                systems.add(len(refs))
                clips.add(sum(bool(m["clipped"]) for m in moves))
                for ref in refs:
                    # Normalize roles and all geometric orientations, preserving the pin.
                    t = ref["tile"]
                    b = np.where(t == body[0], 0, np.where(t == body[1], 1, 2)).astype(np.int8)
                    ref_classes.add(min(_key(_swap(_d4(b, k), (0, 1), s)) for k in range(8) for s in (0, 1)))
                for m in moves:
                    crop_shapes.add(m["patch"].shape)
                    offsets.add(m["offset"])
                    orientations.add(m["orientation"])
                    alignment += m["delta"] == (0, 0)
                    d = m["delta"][m["axis"]]
                    if d:
                        signs.add((m["axis"], int(np.sign(d))))
                # Omitting A's completion or B's movement must alter a training answer.
                if split == "train" and any(m["delta"] != (0, 0) and not m["clipped"] for m in moves):
                    for complete, move in ((False, True), (True, False)):
                        try:
                            wrong = _render(fixed, bg, moves, complete=complete, move=move)
                        except ValueError:
                            continue  # A collision is itself a failed alternative.
                        _require(not np.array_equal(wrong, out), "parent ablation did not change answer")
    _require(len(shapes) >= 45 and len(ref_classes) >= 140, "geometry/reference diversity")
    _require(len(crop_shapes) >= 20 and len(offsets) >= 15, "crop diversity")
    _require(orientations == set(range(8)) and backgrounds == set(range(10)), "orientation/palette coverage")
    _require(systems == {1, 2} and {0, 1, 2} <= clips, "counts/clipping coverage")
    _require(alignment > 0 and len(signs) == 4, "zero displacement or movement directions missing")
    _require(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Metamorphic checks call the input-only solver on novel transformed arrays.
    rng = np.random.default_rng(818)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in range(8):
            palette = rng.permutation(10)
            inp = palette[_d4(pair["input"], k)]
            expected = palette[_d4(pair["output"], k)]
            _require(np.array_equal(solve(inp), expected), "D4/recoloring equivariance")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": 42, "canvas_shapes": len(shapes),
            "reference_classes": len(ref_classes), "crop_shapes": len(crop_shapes),
            "crop_offsets": len(offsets), "aligned_copies": alignment}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
