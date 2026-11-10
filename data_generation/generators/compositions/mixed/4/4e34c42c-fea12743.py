"""Signature registration followed by strict binary union and label projection.

Six framed source panels retain their positions. Matching their repeated 2x2
signatures establishes six virtual, aligned comparison panels. Only the payload
pixels in the chosen operands/result are recolored. No files or parent modules
are read by generation or solving.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a


def _runs(indices):
    """Half-open intervals of consecutive integer indices."""
    result = []
    for x in indices:
        x = int(x)
        if result and result[-1][1] == x:
            result[-1] = (result[-1][0], x + 1)
        else:
            result.append((x, x + 1))
    return result


def _relations(masks):
    found = []
    for i, j in itertools.combinations(range(6), 2):
        a, b = masks[i], masks[j]
        if not a - b or not b - a:
            continue
        union = a | b
        found.extend((i, j, k) for k in range(6)
                     if k not in (i, j) and masks[k] == union)
    return found


def _choose(masks):
    """One mask-level equation, one result slot, equivalent operand ties only."""
    found = _relations(masks)
    equations = {(frozenset((masks[i], masks[j])), masks[k]) for i, j, k in found}
    if len(equations) != 1 or len({k for i, j, k in found}) != 1:
        raise ValueError("no unique strict union equation")
    return min(found)


def _try_choose(masks):
    try:
        return _choose(masks)
    except ValueError:
        return None


def _normalize(mask):
    r0 = min(r for r, c in mask)
    c0 = min(c for r, c in mask)
    return frozenset((r - r0, c - c0) for r, c in mask)


def _parse(input_grid):
    a = _grid(input_grid)
    gutter = int(a[0, 0])
    if not (np.all(a[0] == gutter) and np.all(a[-1] == gutter)
            and np.all(a[:, 0] == gutter) and np.all(a[:, -1] == gutter)):
        raise ValueError("missing complete outer gutter")
    rows = _runs(np.flatnonzero(~np.all(a == gutter, axis=1)))
    cols = _runs(np.flatnonzero(~np.all(a == gutter, axis=0)))
    if (len(rows), len(cols)) not in ((3, 2), (2, 3)):
        raise ValueError("expected six rectangular source panels")
    if len({b - x for x, b in rows}) != 1 or len({b - x for x, b in cols}) != 1:
        raise ValueError("panels must have equal dimensions")
    slots = [(slice(r0, r1), slice(c0, c1))
             for r0, r1 in rows for c0, c1 in cols]
    panels = [a[s] for s in slots]
    background = int(panels[0][0, 0])
    if background == gutter:
        raise ValueError("interior background must differ from gutters")
    if any(not (np.all(p[0] == background) and np.all(p[:, 0] == background))
           or np.any(p == gutter) for p in panels):
        raise ValueError("panel top/left margins must be background")
    colors, counts = np.unique(a, return_counts=True)
    remaining = {int(c): int(n) for c, n in zip(colors, counts)
                 if c not in (background, gutter)}
    # Every panel has one exact key: OO/OR. Payload has >=4 cells per panel,
    # so its global count is >18 and cannot impersonate either marker role.
    if len(remaining) != 3 or sorted(remaining.values())[:2] != [6, 18]:
        raise ValueError("expected six asymmetric two-color signatures")
    result_color = next(c for c, n in remaining.items() if n == 6)
    operand_color = next(c for c, n in remaining.items() if n == 18)
    payloads = [c for c, n in remaining.items() if n > 18]
    if len(payloads) != 1:
        raise ValueError("payload color is ambiguous")
    foreground = payloads[0]
    key_pattern = np.array([[operand_color, operand_color],
                            [operand_color, result_color]])
    anchors, masks, raw_masks = [], [], []
    for p in panels:
        singleton = np.argwhere(p == result_color)
        if singleton.shape != (1, 2) or np.count_nonzero(p == operand_color) != 3:
            raise ValueError("each panel must contain exactly one complete key")
        r, c = (int(x) - 1 for x in singleton[0])
        if r < 1 or c < 1 or not np.array_equal(p[r:r+2, c:c+2], key_pattern):
            raise ValueError("damaged or rotated signature")
        points = frozenset((int(x), int(y)) for x, y in np.argwhere(p == foreground))
        if len(points) < 4:
            raise ValueError("each payload needs at least four cells")
        anchors.append((r, c))
        raw_masks.append(points)
        masks.append(frozenset((x - r, y - c) for x, y in points))
    return {"grid": a, "slots": slots, "panels": panels,
            "arrangement": (len(rows), len(cols)), "background": background,
            "gutter": gutter, "foreground": foreground,
            "operand_color": operand_color, "result_color": result_color,
            "anchors": anchors, "masks": masks, "raw_masks": raw_masks}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, translation and label using this input alone."""
    info = _parse(input_grid)
    i, j, k = _choose(info["masks"])
    output = info["grid"].copy()
    for slot, color in ((i, info["operand_color"]),
                        (j, info["operand_color"]), (k, info["result_color"])):
        block = output[info["slots"][slot]]
        block[info["panels"][slot] == info["foreground"]] = color
    return output


def _orientation(i, j, columns):
    r, c = divmod(i, columns)
    s, d = divmod(j, columns)
    return "horizontal" if r == s else "vertical" if c == d else "diagonal"


def _sample_set(rng, points, count):
    ordered = sorted(points)
    return frozenset(ordered[int(i)] for i in
                     rng.choice(len(ordered), size=count, replace=False))


def _base(rng, height, width, key, style):
    # Keeping the common base away from the top and left edges lets each
    # operand have a different payload bounding box. Exact key alignment then
    # carries information that independent tight cropping would discard.
    allowed = {(r, c) for r in range(1, height) for c in range(1, width)} - key
    if style == "outline":
        points = {(r, c) for r, c in allowed
                  if r in (1, height - 1) or c in (1, width - 1)}
    elif style == "bars":
        rr = int(rng.integers(1, height))
        cc = int(rng.integers(1, width))
        points = {(r, c) for r, c in allowed if r == rr or c == cc}
    elif style == "stair":
        points = {(r, c) for r, c in allowed if abs(r - c) <= 1}
    else:
        points = {p for p in sorted(allowed) if rng.random() < 0.38}
    if len(points) < 3:
        points |= set(_sample_set(rng, allowed - points, 3 - len(points)))
    cap = max(3, min(len(allowed), (height * width - 4) // 2))
    if len(points) > cap:
        points = set(_sample_set(rng, points, cap))
    return frozenset(points)


def _make_masks(rng, height, width, style, duplicate):
    """Construct masks, not whole parent puzzles; all attempts are bounded."""
    for _ in range(180):
        kr = int(rng.integers(height - 1))
        kc = int(rng.integers(width - 1))
        key = {(kr + r, kc + c) for r in range(2) for c in range(2)}
        universe = {(r, c) for r in range(height) for c in range(width)} - key
        top = {(0, c) for c in range(1, width)} - key
        left = {(r, 0) for r in range(1, height)} - key
        if not top or not left:
            continue
        common = _base(rng, height, width, key, style)
        a = common | _sample_set(rng, top, 1)
        b = common | _sample_set(rng, left, 1)
        # More than one unique contribution is optional; the first exclusive
        # top/left pixels always remain exclusive to the intended operand.
        extra = int(rng.integers(0, 4))
        sites_a = {p for p in universe - a - b if p[1] > 0}
        a |= _sample_set(rng, sites_a, min(extra, len(sites_a)))
        sites_b = {p for p in universe - a - b if p[0] > 0}
        b |= _sample_set(rng, sites_b, min(int(rng.integers(0, 4)), len(sites_b)))
        union = a | b
        if len(union) >= len(universe) - 1:
            continue
        # A count-matched decoy and an even larger decoy defeat size ranking.
        near = _sample_set(rng, universe, len(a))
        heavy = _sample_set(rng, universe, min(len(universe), len(union) + 1))
        third = (a if duplicate else
                 _sample_set(rng, universe, int(rng.integers(4, len(universe)))))
        masks = [a, b, union, near, heavy, third]
        if any(not (4 <= len(m) <= len(universe)) for m in masks):
            continue
        if len(set(masks)) != (5 if duplicate else 6):
            continue
        relation = _try_choose(masks)
        if relation != (0, 1, 2):
            continue
        # Reject a family member if independent payload tight crops give the
        # same labels: every returned example witnesses registration necessity.
        if _try_choose([_normalize(m) for m in masks]) == relation:
            continue
        return masks, (kr, kc)
    raise RuntimeError("bounded mask construction exhausted")


def _place_roles(rng, masks, key, columns, orientation):
    candidates = [(i, j) for i, j in itertools.combinations(range(6), 2)
                  if _orientation(i, j, columns) == orientation]
    # Duplicates choose their earliest occurrence only after permutation.
    for _ in range(80):
        order = [int(x) for x in rng.permutation(6)]
        placed = [masks[x] for x in order]
        i, j, k = _choose(placed)
        if (i, j) in candidates:
            return placed, key, (i, j, k)
    raise RuntimeError("bounded role placement exhausted")


def _pair(rng, arrangement, orientation, span, style, duplicate, one_axis):
    rows, columns = arrangement
    max_h = (30 - rows - 1) // rows
    max_w = (30 - columns - 1) // columns
    # A source panel includes at least one blank top/left margin; its payload
    # box may touch the bottom/right edge, but is never clipped.
    height = int(rng.integers(4, min(7, max_h - span - 1) + 1))
    width = int(rng.integers(4, min(7, max_w - span - 1) + 1))
    masks, key = _make_masks(rng, height, width, style, duplicate)
    masks, key, relation = _place_roles(rng, masks, key, columns, orientation)
    gutter, background, foreground, operand_color, result_color = (
        int(c) for c in rng.choice(10, size=5, replace=False))
    panel_h, panel_w = height + span + 1, width + span + 1
    for _ in range(50):
        offsets = rng.integers(1, span + 2, size=(6, 2))
        offsets[0, 0], offsets[-1, 0] = 1, span + 1
        offsets[0, 1], offsets[-1, 1] = span + 1, 1
        if one_axis:
            offsets[:, 0] = 1
        raw = [frozenset((r + int(dr), c + int(dc)) for r, c in m)
               for m, (dr, dc) in zip(masks, offsets)]
        if _try_choose(raw) != relation:
            break
    else:
        raise RuntimeError("bounded displacement construction exhausted")
    shape = rows * (panel_h + 1) + 1, columns * (panel_w + 1) + 1
    input_grid = np.full(shape, gutter, dtype=np.int8)
    output_grid = input_grid.copy()
    first, second, result = relation
    for slot, (mask, (dr, dc)) in enumerate(zip(masks, offsets)):
        rr, cc = divmod(slot, columns)
        top, left = 1 + rr * (panel_h + 1), 1 + cc * (panel_w + 1)
        block = input_grid[top:top+panel_h, left:left+panel_w]
        answer = output_grid[top:top+panel_h, left:left+panel_w]
        block[:] = background
        answer[:] = background
        # These four cells are registration/label-key pixels, not payload.
        ar, ac = int(dr) + key[0], int(dc) + key[1]
        for r, c, color in ((0, 0, operand_color), (0, 1, operand_color),
                            (1, 0, operand_color), (1, 1, result_color)):
            block[ar+r, ac+c] = answer[ar+r, ac+c] = color
        label = (result_color if slot == result else operand_color
                 if slot in (first, second) else foreground)
        for r, c in mask:
            block[r+int(dr), c+int(dc)] = foreground
            answer[r+int(dr), c+int(dc)] = label
    # Ground truth used construction masks/positions. Verify it with the
    # input-only parser; no construction coordinates reach solve().
    if not np.array_equal(solve(input_grid), output_grid):
        raise AssertionError("constructed labels disagree with input reconstruction")
    return {"input": input_grid, "output": output_grid}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    arrangement = ((3, 2), (2, 3))[int(rng.integers(2))]
    orientations = ["horizontal", "vertical", "diagonal"]
    rng.shuffle(orientations)
    styles = ["outline", "bars", "stair", "open"]
    style_offset = int(rng.integers(4))
    ties = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            orientation = (orientations[index % 2] if split == "train" else
                           orientations[2] if index == 0 else str(rng.choice(orientations)))
            duplicate = ties and ((split == "train" and index == 1)
                                  or (index > 1 and rng.random() < 0.4)
                                  or (split == "test" and rng.random() < 0.6))
            for _ in range(16):
                try:
                    pair = _pair(rng, arrangement, orientation,
                                 1 if split == "train" else 2,
                                 styles[(style_offset + index) % 4], duplicate,
                                 split == "train" and index == 0)
                except RuntimeError:
                    continue
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    relation = _choose(info["masks"])
    if out.shape != inp.shape or not np.array_equal(solve(inp), out):
        raise AssertionError("incorrect output")
    i, j, k = relation
    masks = info["masks"]
    assert masks[i] | masks[j] == masks[k]
    assert masks[i] - masks[j] and masks[j] - masks[i]
    assert masks[i] & masks[j], "generated operands must overlap, distinguishing XOR"
    assert (masks[i] ^ masks[j]) != masks[k]
    assert _try_choose(info["raw_masks"]) != relation, "registration ablation did not change labels"
    assert _try_choose([_normalize(m) for m in masks]) != relation, "tight-crop shortcut survived"
    assert max(len(m) for m in masks) > len(masks[k]), "largest-mask shortcut not excluded"
    assert any(len(masks[s]) == len(masks[t]) and masks[s] != masks[t]
               for s in range(6) if s not in relation for t in (i, j)), "missing count-matched decoy"
    changed = inp != out
    selected = np.zeros(inp.shape, dtype=bool)
    for slot in relation:
        selected[info["slots"][slot]] = info["panels"][slot] == info["foreground"]
    assert np.array_equal(changed, selected), "keys, gutters, holes, or distractors changed"
    assert np.count_nonzero(out == info["foreground"]) > 0
    info["relation"] = relation
    info["raw_relation_count"] = len(_relations(masks))
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = (a.shape, a.astype(np.uint8).tobytes())
            assert key not in seen, "repeated input"
            seen.add(key)
            infos[split].append(_analyze_pair(pair))
    assert len({i["arrangement"] for group in infos.values() for i in group}) == 1
    train_orientations = {_orientation(*i["relation"][:2], i["arrangement"][1])
                          for i in infos["train"]}
    assert len(train_orientations) >= 2, "demonstrations need distinct role arrangements"
    if any(i["raw_relation_count"] > 1 for i in infos["test"]):
        assert any(i["raw_relation_count"] > 1 for i in infos["train"]), "undemonstrated duplicate tie"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain hashes/statistics rather than complete puzzles."""
    digests, shapes, arrangements, relations, normalized = set(), set(), set(), set(), set()
    palettes, sizes, contributions = set(), set(), set()
    tied, overlap_total = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                masks = info["masks"]
                i, j, k = info["relation"]
                shapes.add(pair["input"].shape)
                arrangements.add(info["arrangement"])
                relations.add(info["relation"])
                palettes.add(tuple(info[r] for r in ("gutter", "background", "foreground",
                                                      "operand_color", "result_color")))
                sizes.update(len(m) for m in masks)
                contributions.add((len(masks[i] - masks[j]), len(masks[j] - masks[i])))
                tied += info["raw_relation_count"] > 1
                overlap_total += len(masks[i] & masks[j])
                mapping = {}
                pattern = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                normalized.add(hashlib.sha256(pattern).hexdigest())
                # Lists and arrays follow the same input-only path, and solving
                # must not mutate its argument.
                before = pair["input"].copy()
                assert np.array_equal(solve(before.tolist()), pair["output"])
                assert np.array_equal(before, pair["input"])
        # Signature-anchor displacement is the meaningful extrapolation.
        for info in [_parse(p["input"]) for p in puzzle["test"]]:
            assert all(max(p[a] for p in info["anchors"]) - min(p[a] for p in info["anchors"]) == 2
                       for a in (0, 1))
    assert len(digests) == 42 and len(normalized) == 210
    assert arrangements == {(3, 2), (2, 3)}
    assert len(shapes) >= 15 and len(relations) >= 25 and len(palettes) >= 100
    assert tied >= 10 and len(sizes) >= 15 and len(contributions) >= 10 and overlap_total > 500
    assert _digest(generate_puzzle(203, 7, 3)) == _digest(generate_puzzle(203, 7, 3))
    # Verify maximal counts once; the supplied verifier exercises all legal counts.
    boundary = generate_puzzle(204, 10, 4)
    assert len(boundary["train"]) == 10 and len(boundary["test"]) == 4
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.2, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = generate_puzzle(0)["train"][0]
    broken = {r: a.copy() for r, a in pair.items()}
    changed = np.argwhere(broken["input"] != broken["output"])[0]
    broken["output"][tuple(changed)] = broken["input"][tuple(changed)]
    try:
        _analyze_pair(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing label accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "grid_shapes": len(shapes), "normalized_inputs": len(normalized),
            "role_triples": len(relations), "palettes": len(palettes),
            "foreground_counts": len(sizes), "unique_contribution_pairs": len(contributions),
            "duplicate_equation_pairs": tied}
