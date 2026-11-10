"""Locate a cropped view, then mark the complete pluses it cuts.

All solving information is visible in each input. No parent code, saved bitmap,
training labels, seed, cache, or construction metadata enters solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 692478622, 904666366, 830012799, 248141877, 356552196,
               683293917, 875533294, 169226942, 160601543, 862766210,
               469357099, 214564056, 921447488, 785312954, 560118749,
               612514680, 658014796, 868024264, 140852791, 415285268,
               143660766, 201206987, 450257455, 355274646, 607647831,
               690838715, 816850513, 339543372, 284808001, 550118846,
               656022048, 415737677, 717510640, 70481099, 357990805,
               337270192, 392212538, 409245976, 777294954, 456256009,
               388713176)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _d4(a, index):
    """Four rotations, followed by four rotations of a left/right reflection."""
    return np.rot90(a if index < 4 else np.fliplr(a), index % 4).copy()


def _compose(a, b):
    probe = np.arange(6).reshape(2, 3)
    target = _d4(_d4(probe, b), a)
    return next(t for t in range(8) if np.array_equal(_d4(probe, t), target))


def _erode(a):
    b = np.zeros_like(a, dtype=bool)
    b[1:-1, 1:-1] = (a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1]
                      & a[1:-1, :-2] & a[1:-1, 2:])
    return b


def _arms(centers):
    a = centers.copy()
    a[1:] |= centers[:-1]
    a[:-1] |= centers[1:]
    a[:, 1:] |= centers[:, :-1]
    a[:, :-1] |= centers[:, 1:]
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    component.append(point)
                    stack.append(point)
        result.append(component)
    return result


def _parse(inp):
    """Find the two same-ink rectangular frames; larger interior is the atlas."""
    parses = []
    for color in map(int, np.unique(inp)):
        if color == 0:
            continue
        components = _components(inp == color)
        if len(components) != 2:
            continue
        boxes, interiors = [], []
        good = True
        for component in components:
            rr, cc = zip(*component)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            if min(r1 - r0, c1 - c0) < 4:
                good = False
                break
            expected = {(r, c) for r in range(r0, r1 + 1)
                        for c in range(c0, c1 + 1)
                        if r in (r0, r1) or c in (c0, c1)}
            if set(component) != expected:
                good = False
                break
            interior = inp[r0 + 1:r1, c0 + 1:c1]
            if len(np.unique(interior)) != 2 or np.any(interior == color):
                good = False
                break
            boxes.append((r0, r1, c0, c1))
            interiors.append(interior.copy())
        if not good:
            continue
        exterior = np.ones(inp.shape, dtype=bool)
        for r0, r1, c0, c1 in boxes:
            if not exterior[r0:r1 + 1, c0:c1 + 1].all():
                good = False
            exterior[r0:r1 + 1, c0:c1 + 1] = False
        if not good or np.any(inp[exterior] != 0):
            continue
        if interiors[0].size == interiors[1].size:
            continue
        atlas, patch = sorted(interiors, key=lambda a: -a.size)
        colors, counts = np.unique(atlas, return_counts=True)
        if counts[0] == counts[1]:
            continue
        foreground = int(colors[np.argmin(counts)])
        parses.append((atlas, patch, foreground, color))
    if len(parses) != 1:
        raise ValueError("input must encode one unambiguous atlas/view frame pair")
    return parses[0]


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero((windows[:, :, 0] == patch[0]).all(axis=-1))
    for row in range(1, h):
        if not len(rr):
            break
        keep = (full[rr[:, None] + row, cc[:, None] + np.arange(w)]
                == patch[row]).all(axis=-1)
        rr, cc = rr[keep], cc[keep]
    return list(zip(map(int, rr), map(int, cc)))


def _matches(mask, patch):
    """Check both role bijections, all D4 operations, and all valid offsets."""
    colors = np.unique(patch)
    if len(colors) != 2:
        return []
    normalized = patch == colors[1]
    matches = []
    for t in range(8):
        full = _d4(mask, t)
        for swap in (0, 1):
            query = normalized if not swap else ~normalized
            for r, c in _locations(full, query):
                matches.append((t, swap, r, c))
    return matches


def _support(shape, patch_shape, match):
    t, _, r, c = match
    # Transform cell identities, so the crop's support returns to atlas coordinates.
    identities = _d4(np.arange(np.prod(shape)).reshape(shape), t)
    h, w = patch_shape
    support = np.zeros(shape, dtype=bool)
    support.flat[identities[r:r + h, c:c + w].ravel()] = True
    return support


def _analyze(inp):
    atlas, patch, fg, pen = _parse(inp)
    mask = atlas == fg
    matches = _matches(mask, patch)
    if len(matches) != 1:
        raise ValueError("view must have one transform/role/location match")
    support = _support(atlas.shape, patch.shape, matches[0])
    if support.all():
        raise ValueError("the view must be a proper crop")
    centers = _erode(mask)
    selected = centers & support & ~_erode(support)
    return atlas, patch, fg, pen, support, centers, selected, matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the atlas, marking full pluses whose centers are cut by the view."""
    atlas, _, _, pen, _, _, selected, _ = _analyze(_grid(input_grid))
    answer = atlas.copy()
    answer[_arms(selected)] = pen
    return answer


def _window_info(mask, r, c, h, w):
    support = np.zeros_like(mask)
    support[r:r + h, c:c + w] = True
    centers = _erode(mask)
    selected = centers & support & ~_erode(support)
    union = _arms(selected)
    # Directly simulate destructive recoloring of only the eligible crosses.
    remaining = mask.copy()
    destructive = np.zeros_like(mask)
    for a, b in np.argwhere(selected):
        points = [(a, b)] + [(a + dr, b + dc) for dr, dc in _DIRS]
        if all(remaining[p] for p in points):
            for p in points:
                remaining[p] = False
                destructive[p] = True
    corners = {(r, c), (r, c + w - 1), (r + h - 1, c),
               (r + h - 1, c + w - 1)}
    return dict(selected=int(selected.sum()), interior=int((centers & _erode(support)).sum()),
                outside_touch=bool(np.any(centers & ~support & _arms(support))),
                interior_unique=bool(np.any(_arms(centers & _erode(support)) & ~union)),
                outside_unique=bool(np.any(_arms(centers & ~support & _arms(support)) & ~union)),
                overlap=not np.array_equal(union, destructive),
                corner=any(selected[p] for p in corners),
                attached=bool(np.any(mask & ~union & _arms(union))),
                restored=int((union & ~support).sum()),
                sides=int(selected[r, c:c + w].any())
                      + int(selected[r + h - 1, c:c + w].any())
                      + int(selected[r:r + h, c].any())
                      + int(selected[r:r + h, c + w - 1].any()))


def _stamp(mask, points, cap):
    points = {(int(r), int(c)) for r, c in points
              if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]}
    if int(mask.sum()) + sum(not mask[p] for p in points) <= cap:
        for p in points:
            mask[p] = True


def _plus(r, c):
    return [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]


def _draw_reference(rng, shape):
    """Structured ink only: crosses, branches, runs, loops, blocks and near misses."""
    H, W = shape
    mask = np.zeros(shape, dtype=bool)
    cap = int(.46 * H * W)
    # Two adjacent crosses guarantee an overlap ingredient, in a variable location.
    r = int(rng.integers(2, H - 3))
    c = int(rng.integers(2, W - 4))
    for p in _plus(r, c) + _plus(r, c + 1):
        mask[p] = True
    for _ in range(int(rng.integers(5, 12))):
        r, c = int(rng.integers(1, H - 1)), int(rng.integers(1, W - 1))
        _stamp(mask, _plus(r, c), cap)
    for _ in range(int(rng.integers(5, 13))):
        r, c = int(rng.integers(H)), int(rng.integers(W))
        kind = int(rng.integers(6))
        a, b = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        if kind == 0:  # bent run
            points = [(r + i, c) for i in range(a)] + [(r + a - 1, c + j) for j in range(b)]
        elif kind == 1:  # hollow loop
            points = [(r + i, c + j) for i in range(a + 1) for j in range(b + 1)
                      if i in (0, a) or j in (0, b)]
        elif kind == 2:  # thick short bar
            points = [(r + i, c + j) for i in range(2) for j in range(b)]
        elif kind == 3:  # T missing its fourth arm
            points = [(r, c), (r - 1, c), (r, c - 1), (r, c + 1)]
        elif kind == 4:  # stepped diagonal
            points = [(r + i // 2, c + (i + 1) // 2) for i in range(a + b)]
        else:  # unequal branches
            points = [(r, c + j) for j in range(b + 2)]
            points += [(r + i, c + j) for j in (1, b) for i in range(a)]
        _stamp(mask, points, cap)
    # Occasionally leave a clipped plus on the atlas edge. It is not a center.
    if rng.random() < .5:
        _stamp(mask, _plus(0, int(rng.integers(1, W - 1))), cap)
    return _d4(mask, int(rng.integers(8)))


def _geometry_candidates(rng, mask, thin):
    """Bounded random proposals, anchored to actual plus centers."""
    H, W = mask.shape
    centers = list(map(tuple, np.argwhere(_erode(mask))))
    if not centers:
        return
    seen = set()
    for _ in range(900):
        if thin:
            h, w = int(rng.integers(3, 5)), int(rng.integers(5, min(W, 12) + 1))
            if rng.random() < .5:
                h, w = w, h
        else:
            h = int(rng.integers(5, min(H - 1, 10) + 1))
            w = int(rng.integers(5, min(W - 1, 10) + 1))
        if h > H or w > W:
            continue
        a, b = centers[int(rng.integers(len(centers)))]
        edge = int(rng.integers(4))
        corner = rng.random() < .4
        if edge < 2:
            r = a if edge == 0 else a - h + 1
            c = b - (int(rng.choice((0, w - 1))) if corner else int(rng.integers(w)))
        else:
            c = b if edge == 2 else b - w + 1
            r = a - (int(rng.choice((0, h - 1))) if corner else int(rng.integers(h)))
        if not (0 <= r <= H - h and 0 <= c <= W - w):
            continue
        window = (int(r), int(c), h, w)
        if window in seen:
            continue
        seen.add(window)
        # Both axis orders must fit, so demonstration operations can be chosen
        # independently of crop shape and canvas packing.
        if not all(H + qh + 5 <= 30 or W + qw + 5 <= 30
                   for qh, qw in ((h, w), (w, h))):
            continue
        yield window


def _make_banks(rng, mask):
    banks = {"edge": [], "corner": [], "thin": []}
    for thin in (False, True):
        for r, c, h, w in _geometry_candidates(rng, mask, thin):
            info = _window_info(mask, r, c, h, w)
            if not info["selected"]:
                continue
            if thin:
                wanted = ["thin"]
            else:
                # Both anchor demonstrations defeat full-map, whole-component,
                # fully-visible-only and arm-intersection shortcuts.
                if not (info["interior_unique"] and info["outside_unique"] and info["attached"]):
                    continue
                wanted = (["edge"] if info["overlap"] else [])
                if info["corner"]:
                    wanted.append("corner")
            wanted = [name for name in wanted if len(banks[name]) < 8]
            if not wanted:
                continue
            patch = mask[r:r + h, c:c + w].astype(np.int8)
            if len(_matches(mask, patch)) != 1:
                continue
            for name in wanted:
                banks[name].append((r, c, h, w))
            if thin and len(banks["thin"]) >= 8:
                break
            if not thin and min(len(banks["edge"]), len(banks["corner"])) >= 8:
                break
    return banks


def _reference(rng):
    H = int(rng.integers(11, 19))
    W = H if rng.random() < .35 else int(rng.integers(11, 20))
    for _ in range(48):
        mask = _draw_reference(rng, (H, W))
        if not .15 <= mask.mean() <= .46:
            continue
        if len({_key(_d4(mask, t)) for t in range(8)}) < 8:
            continue
        banks = _make_banks(rng, mask)
        if all(banks.values()):
            return mask, banks
    raise RuntimeError("bounded reference search exhausted")


def _frame(a, pen):
    framed = np.full((a.shape[0] + 2, a.shape[1] + 2), pen, dtype=np.int8)
    framed[1:-1, 1:-1] = a
    return framed


def _pack(rng, atlas, patch, pen):
    panels = [_frame(atlas, pen), _frame(patch, pen)]
    if rng.random() < .5:
        panels.reverse()
    a, b = panels
    choices = []
    for axis in (0, 1):
        for gap in (1, 2, 3):
            total = a.shape[axis] + b.shape[axis] + gap
            other = max(a.shape[1 - axis], b.shape[1 - axis])
            if total <= 30 and other <= 30:
                choices.append((axis, gap, total, other))
    if not choices:
        return None
    axis, gap, total, other = choices[int(rng.integers(len(choices)))]
    margin = int(rng.integers(0, min(2, (30 - total) // 2) + 1))
    cross_margin = int(rng.integers(0, min(2, (30 - other) // 2) + 1))
    shape = ((total + 2 * margin, other + 2 * cross_margin) if axis == 0
             else (other + 2 * cross_margin, total + 2 * margin))
    canvas = np.zeros(shape, dtype=np.int8)
    position = margin
    for panel in panels:
        across = int(rng.integers(cross_margin, shape[1 - axis] - panel.shape[1 - axis] - cross_margin + 1))
        r, c = (position, across) if axis == 0 else (across, position)
        canvas[r:r + panel.shape[0], c:c + panel.shape[1]] = panel
        position += panel.shape[axis] + gap
    return canvas


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    reference, banks = _reference(rng)
    pen = int(rng.integers(1, 10))
    available = [c for c in range(10) if c != pen]
    palette = tuple(map(int, rng.choice(available, 2, replace=False)))
    palette_mode = int(rng.integers(3))  # stable, role swaps, independent renaming
    # A quarter turn and a reflection demonstrate generators of the full D4
    # family, even with only two examples. The explicit atlas fixes identity.
    anchor_operations = [int(rng.choice((1, 3))), int(rng.integers(4, 8))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            group = ("thin" if split == "test" else "edge" if index == 0
                     else "corner" if index == 1 else str(rng.choice(("edge", "corner", "thin"))))
            for attempt in range(160):
                r, c, h, w = banks[group][int(rng.integers(len(banks[group])))]
                a_t = int(rng.integers(8))
                relative = (anchor_operations[index] if split == "train" and index < 2
                            else int(rng.integers(8)))
                p_t = _compose(relative, a_t)
                atlas_mask = _d4(reference, a_t)
                patch_mask = _d4(reference[r:r + h, c:c + w], p_t)
                a_colors = palette
                p_colors = palette
                if palette_mode == 1:
                    if rng.random() < .5:
                        a_colors = palette[::-1]
                    if rng.random() < .5:
                        p_colors = palette[::-1]
                elif palette_mode == 2:
                    a_colors = tuple(map(int, rng.choice(available, 2, replace=False)))
                    p_colors = tuple(map(int, rng.choice(available, 2, replace=False)))
                atlas = np.where(atlas_mask, a_colors[1], a_colors[0]).astype(np.int8)
                patch = np.where(patch_mask, p_colors[1], p_colors[0]).astype(np.int8)
                inp = _pack(rng, atlas, patch, pen)
                if inp is None or _key(inp) in seen:
                    continue
                # Labels come from the planned spatial support, not solve().
                support = np.zeros_like(reference)
                support[r:r + h, c:c + w] = True
                selected = _erode(reference) & support & ~_erode(support)
                out = atlas.copy()
                out[_d4(_arms(selected), a_t)] = pen
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("bounded layout/uniqueness search exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _scan_expected(atlas, foreground, pen, support):
    """Coordinate-based validation of the intended rule; no erosion/dilation."""
    H, W = atlas.shape
    answer = atlas.copy()
    eligible, changed = [], set()
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            cells = [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]
            if all(atlas[p] == foreground for p in cells):
                if support[r, c] and any(not support[p] for p in cells):
                    eligible.append((r, c))
                    changed.update(cells)
    for point in changed:
        answer[point] = pen
    return answer, eligible, changed


def _features(inp):
    atlas, patch, fg, pen, support, centers, selected, match = _analyze(inp)
    rr, cc = np.nonzero(support)
    r, c = int(rr.min()), int(cc.min())
    h, w = int(np.ptp(rr)) + 1, int(np.ptp(cc)) + 1
    return dict(_window_info(atlas == fg, r, c, h, w),
                atlas_shape=atlas.shape, view_shape=patch.shape, area=h * w / atlas.size,
                orientation=match[0], pen=pen, foreground=fg,
                palette=tuple(map(int, np.unique(atlas))),
                query_palette=tuple(map(int, np.unique(patch))),
                offset=(r, c))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, canonical, pens = set(), set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            atlas, patch, fg, pen, support, centers, selected, _ = _analyze(inp)
            canonical.add(min(_key(_d4((atlas == fg).astype(np.int8), t)) for t in range(8)))
            pens.add(pen)
            assert .15 <= np.mean(atlas == fg) <= .46
            expected, eligible, changed = _scan_expected(atlas, fg, pen, support)
            assert eligible and changed, "generated examples must show a cut junction"
            assert np.array_equal(expected, out), "incorrect cut-plus labeling"
            assert np.array_equal(solve(inp), out)
            assert out.shape == atlas.shape
            assert int(np.count_nonzero(out != atlas)) == len(changed)
            assert any(not support[p] for p in changed), "restore the invisible arms too"
            features[split].append(_features(inp))
    assert len(canonical) == len(pens) == 1, "one shared reference and pen per puzzle"
    train, test = features["train"], features["test"]
    assert any(f["interior_unique"] and f["outside_unique"] and f["attached"] for f in train)
    assert any(f["overlap"] for f in train), "simultaneous overlap witness required"
    assert any(f["corner"] for f in train), "two-arm corner cuts must be demonstrated"
    assert any(f["orientation"] in (1, 3) for f in train), "show a quarter turn"
    assert any(f["orientation"] >= 4 for f in train), "show a reflection"
    assert any(min(f["view_shape"]) >= 5 for f in train)
    assert all(3 <= min(f["view_shape"]) <= 4 for f in test)


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 puzzles streamed in memory; no files, traces, models or unbounded loops."""
    signatures, references, shapes, offsets, palettes, orientations = set(), set(), set(), set(), set(), set()
    changed_counts, center_counts, pens, foregrounds = set(), set(), set(), set()
    counts = dict(pairs=0, overlap=0, corner=0, multi_side=0, restored=0)
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        first = _analyze(puzzle["train"][0]["input"])
        references.add(min(_key(_d4((first[0] == first[2]).astype(np.int8), t)) for t in range(8)))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(pair["input"].shape)
                offsets.add(f["offset"])
                palettes.add((f["palette"], f["query_palette"]))
                orientations.add(f["orientation"])
                center_counts.add(f["selected"])
                pens.add(f["pen"])
                foregrounds.add(f["foreground"])
                atlas = _parse(pair["input"])[0]
                changed_counts.add(int(np.count_nonzero(pair["output"] != atlas)))
                counts["pairs"] += 1
                counts["overlap"] += int(f["overlap"])
                counts["corner"] += int(f["corner"])
                counts["multi_side"] += int(f["sides"] >= 3)
                counts["restored"] += f["restored"]
    assert len(signatures) == len(references) == 42
    assert len(shapes) >= 30 and len(offsets) >= 25 and len(palettes) >= 20
    assert orientations == set(range(8))
    assert len(center_counts) >= 4 and len(changed_counts) >= 10
    assert len(pens) >= 7 and len(foregrounds) >= 8
    assert counts["overlap"] >= 42 and counts["corner"] >= 42 and counts["multi_side"] >= 5
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(19, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Every possible 3x3 mask: production motif recognition agrees with direct scan.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=np.int8).reshape(3, 3)
        support = np.zeros((3, 3), dtype=bool)
        support[1:, 1:] = True
        expected, _, _ = _scan_expected(mask, 1, 8, support)
        actual = mask.copy()
        actual[_arms(_erode(mask == 1) & support & ~_erode(support))] = 8
        assert np.array_equal(expected, actual)
    # A selected plus on a crop corner restores two arms; a T at the atlas edge
    # is never promoted, and an outside center remains unselected even if its arm enters.
    mask = np.zeros((8, 10), dtype=np.int8)
    for point in _plus(3, 3) + _plus(2, 6):
        mask[point] = 1
    mask[0, 1:4] = 1
    mask[1, 2] = 1
    support = np.zeros_like(mask, dtype=bool)
    support[3:7, 3:9] = True
    answer, eligible, cells = _scan_expected(mask, 1, 8, support)
    assert eligible == [(3, 3)] and len(cells) == 5 and answer[2, 6] == 1
    assert answer[0, 2] == 1 and answer[2, 3] == answer[3, 2] == 8
    # Content corruption, including an otherwise valid atlas color, must be rejected.
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed output cell")
    return dict(seeds=42, distinct_references=len(references), input_shapes=len(shapes),
                offsets=len(offsets), palette_pairs=len(palettes), orientations=sorted(orientations),
                selected_center_counts=sorted(center_counts), changed_counts=sorted(changed_counts),
                **counts)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
