"""A locally demonstrated rectangle-interior operation in a missing-panel analogy.

No parent generator is imported. Generation constructs logical panels; solve
segments the input pixels and infers the neutral/highlight roles locally.
All diagnostics are bounded and in memory; this module writes no files.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _bands(mask):
    """All horizontally maximal solid runs of every row band, even thin ones.

    Every globally largest rectangle is in this list: a missing horizontal
    extension would increase its area. Bounds are (top, left, bottom, right),
    with bottom/right exclusive. No square or aspect-ratio preference.
    """
    h, w = mask.shape
    result = []
    for top in range(h):
        columns = np.ones(w, dtype=bool)
        for bottom in range(top + 1, h + 1):
            columns &= mask[bottom - 1]
            changes = np.diff(np.r_[False, columns, False].astype(np.int8))
            for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                result.append((top, int(left), bottom, int(right)))
    return result


def _area(box):
    r, c, b, d = box
    return (b - r) * (d - c)


def _largest(panel, neutral):
    boxes = _bands(panel == neutral)
    if not boxes:
        raise ValueError("no neutral rectangle")
    maximum = max(map(_area, boxes))
    winners = [b for b in boxes if _area(b) == maximum]
    if len(winners) != 1:
        raise ValueError("greatest-area neutral rectangle is tied")
    box = winners[0]
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        raise ValueError("winning rectangle needs a nonempty strict interior")
    return box


def _maximal_boxes(panel, neutral):
    """Inclusion-maximal boxes, for diagnostics, not an alternative selection."""
    mask = panel == neutral
    h, _ = mask.shape
    return [b for b in _bands(mask)
            if (b[0] == 0 or not mask[b[0] - 1, b[1]:b[3]].all())
            and (b[2] == h or not mask[b[2], b[1]:b[3]].all())]


def _interior(shape, box):
    r, c, b, d = box
    mask = np.zeros(shape, dtype=bool)
    mask[r + 1:b - 1, c + 1:d - 1] = True
    return mask


def _paint(panel, box, highlight):
    out = panel.copy()
    r, c, b, d = box
    out[r + 1:b - 1, c + 1:d - 1] = highlight
    return out


def _inside(a, b):
    return b[0] < a[0] and b[1] < a[1] and a[2] < b[2] and a[3] < b[3]


def _disjoint(a, b, gap=0):
    return (a[2] + gap <= b[0] or b[2] + gap <= a[0]
            or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _outlines(grid):
    """Find isolated monochrome outline components by geometry, not color IDs.

    Structural colors are reserved for outlines in this family. Outlines may
    share a color when they remain disconnected. Internal neutral rings can
    also be detected here; nesting determines which outlines are panel frames.
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    found = []
    for row in range(h):
        for col in range(w):
            if visited[row, col]:
                continue
            color = int(grid[row, col])
            stack, cells = [(row, col)], []
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= y < h and 0 <= x < w and not visited[y, x]
                            and grid[y, x] == color):
                        visited[y, x] = True
                        stack.append((y, x))
            if len(cells) < 16:
                continue
            r0 = min(p[0] for p in cells)
            c0 = min(p[1] for p in cells)
            r1 = max(p[0] for p in cells) + 1
            c1 = max(p[1] for p in cells) + 1
            rh, rw = r1 - r0, c1 - c0
            if (min(rh, rw) >= 5 and len(cells) == 2 * (rh + rw) - 4
                    and all(r in (r0, r1 - 1) or c in (c0, c1 - 1) for r, c in cells)):
                found.append((r0, c0, r1, c1, color))
    return found


def _crop(grid, frame):
    r, c, b, d = frame[:4]
    return grid[r + 1:b - 1, c + 1:d - 1].copy()


def _relation(reference, neutral):
    """Determine direction and highlight from visible source-to-partner changes."""
    candidates = []
    for source_slot in (0, 1):
        source, partner = reference[source_slot], reference[1 - source_slot]
        changed = source != partner
        if not changed.any() or not np.all(source[changed] == neutral):
            continue
        colors = np.unique(partner[changed])
        if len(colors) != 1:
            continue
        highlight = int(colors[0])
        if highlight == neutral or np.any(source == highlight):
            continue
        try:
            box = _largest(source, neutral)
        except ValueError:
            continue
        if np.array_equal(changed, _interior(source.shape, box)):
            candidates.append((source, partner, source_slot, box, highlight))
    if len(candidates) != 1:
        raise ValueError("complete pair does not show one unique interior-paint relation")
    return candidates[0]


def _interpret(input_grid):
    grid = _grid(input_grid)
    outlines = _outlines(grid)
    groups = []
    for outer in outlines:
        descendants = [p for p in outlines if _inside(p, outer)]
        children = [p for p in descendants
                    if not any(_inside(p, q) for q in descendants)]
        if len(children) != 2 or not _disjoint(*children):
            continue
        if (children[0][2] - children[0][0], children[0][3] - children[0][1]) != (
                children[1][2] - children[1][0], children[1][3] - children[1][1]):
            continue
        # Reading order is recorded for diagnostics only, never for direction.
        children.sort(key=lambda b: (b[0], b[1]))
        groups.append((outer, children))
    scenes = []
    for g0, g1 in combinations(groups, 2):
        if not _disjoint(g0[0], g1[0]):
            continue
        boxes = [g0[1], g1[1]]
        panels = [[_crop(grid, f) for f in members] for members in boxes]
        if len({p.shape for pair in panels for p in pair}) != 1:
            continue
        blanks = [(g, s, int(p[0, 0])) for g in range(2)
                  for s, p in enumerate(panels[g]) if np.all(p == p[0, 0])]
        if len(blanks) != 1:
            continue
        target_group, blank_slot, neutral = blanks[0]
        try:
            source, partner, source_slot, ref_box, highlight = _relation(
                panels[1 - target_group], neutral)
            target = panels[target_group][1 - blank_slot]
            if np.any(target == highlight):
                continue
            target_box = _largest(target, neutral)
        except ValueError:
            continue
        scenes.append(dict(grid=grid, groups=[g0[0], g1[0]], frames=boxes,
                           source=source, partner=partner, source_slot=source_slot,
                           reference_group=1 - target_group, target_group=target_group,
                           blank_slot=blank_slot, neutral=neutral, highlight=highlight,
                           ref_box=ref_box, target_box=target_box, target=target,
                           answer=_paint(target, target_box, highlight)))
    if len(scenes) != 1:
        raise ValueError(f"expected one supported four-panel scene; found {len(scenes)}")
    return scenes[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the missing companion interior using only the given input."""
    return _interpret(input_grid)["answer"].copy()


def _geometry(rng, shape, decoys):
    h, w = shape
    for _ in range(96):
        high = 5 if decoys == 2 else 7
        rh = int(rng.integers(3 if decoys < 2 else 4, min(high, h - 1) + 1))
        rw = int(rng.integers(3 if decoys < 2 else 4, min(high, w - 1) + 1))
        if decoys and rh * rw < 12:
            continue
        r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        boxes = [(r, c, r + rh, c + rw)]
        for _ in range(decoys):
            dimensions = [(a, b) for a in range(3, min(h, 6) + 1)
                          for b in range(3, min(w, 6) + 1) if a * b <= .8 * rh * rw]
            if not dimensions:
                break
            dh, dw = dimensions[int(rng.integers(len(dimensions)))]
            places = [(y, x, y + dh, x + dw) for y in range(h - dh + 1)
                      for x in range(w - dw + 1)
                      if all(_disjoint((y, x, y + dh, x + dw), b, 1) for b in boxes)]
            if not places:
                break
            boxes.append(places[int(rng.integers(len(places)))])
        if len(boxes) == decoys + 1:
            return boxes
    # Guaranteed geometry for the same rule, not a new operation or a retry loop.
    if decoys:
        boxes = [(0, 0, 4, 4), (h - 3, w - 3, h, w)]
        if decoys == 2:
            boxes.append((0, w - 3, 3, w))
        return boxes
    return [(1, 1, 4, 5)]


def _ring(shape, box):
    r, c, b, d = box
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, r - 1):min(shape[0], b + 1), max(0, c - 1):min(shape[1], d + 1)] = True
    mask[r:b, c:d] = False
    return mask


def _constructive_field(shape, neutral, textures, decoys):
    """Guaranteed spare geometry for a target whose random bounds all repeated."""
    h, w = shape
    extent = 4 if decoys else 3
    boxes = [(0, 0, extent, 4)]
    if decoys:
        boxes.append((h - 3, w - 3, h, w))
    if decoys == 2:
        boxes.append((0, w - 3, 3, w))
    rr, cc = np.indices(shape)
    grid = np.asarray(textures, dtype=np.int8)[(rr + cc) % len(textures)]
    for r, c, b, d in boxes:
        grid[r:b, c:d] = neutral
    grid[extent, 0] = neutral
    return grid


def _field(rng, shape, neutral, textures, decoys):
    boxes = _geometry(rng, shape, decoys)
    target = boxes[0]
    whole = np.zeros(shape, dtype=bool)
    for r, c, b, d in boxes:
        whole[r:b, c:d] = True
    ring = _ring(shape, target)
    for attempt in range(65):
        # Final attempt is a guaranteed field: no unplanted neutral runs,
        # except one partial side contact that cannot extend a whole boundary.
        if attempt == 64:
            rr, cc = np.indices(shape)
            grid = np.asarray(textures, dtype=np.int8)[(rr + cc) % len(textures)]
        else:
            weights = rng.dirichlet(np.full(len(textures), 1.5))
            grid = rng.choice(textures, size=shape, p=weights).astype(np.int8)
            mask = rng.random(shape) < rng.uniform(.16, .48)
            if rng.random() < .4:
                # Short correlated runs, not a clean surrounding halo.
                use = rng.random(shape) < .18
                mask[use] = np.roll(mask, 1, axis=int(rng.integers(2)))[use]
            grid[mask] = neutral
        grid[whole] = neutral
        if not np.any((grid == neutral) & ring):
            contacts = np.argwhere(ring & ~whole)
            if len(contacts):
                y, x = contacts[int(rng.integers(len(contacts)))]
                grid[y, x] = neutral
        try:
            if _largest(grid, neutral) != target:
                continue
        except ValueError:
            continue
        # Keep disjoint distractors clearly smaller; overlapping subrectangles
        # of the winner are not competing objects.
        outside = grid.copy()
        r, c, b, d = target
        outside[r:b, c:d] = -1
        rivals = _bands(outside == neutral)
        if rivals and max(map(_area, rivals)) > .8 * _area(target):
            continue
        return grid, target
    raise RuntimeError("bounded field construction failed")


def _outline(grid, box, color):
    r, c, b, d = box
    grid[r, c:d] = grid[b - 1, c:d] = color
    grid[r:b, c] = grid[r:b, d - 1] = color


def _layout(rng, shape, contents, colors):
    h, w = shape
    ph, pw = h + 2, w + 2
    gh_min, gw_min = ph + 2, 2 * pw + 3
    height = int(rng.integers(2 * gh_min + 1, 31))
    width = int(rng.integers(gw_min, 31))
    slack = rng.multinomial(height - (2 * gh_min + 1), [0.2] * 5)
    heights = [gh_min + int(slack[0]), gh_min + int(slack[1])]
    tops = [int(slack[2]), int(slack[2]) + heights[0] + 1 + int(slack[3])]
    # Wallpaper uses content colors, excluding the locally inferred highlight
    # and all reserved frame colors. Short equal-color runs are allowed.
    palette = np.asarray([colors[0], *colors[6:]], dtype=np.int8)
    rr, cc = np.indices((height, width))
    tile = rng.choice(palette, size=(int(rng.integers(2, 5)), int(rng.integers(2, 5))))
    grid = tile[rr % tile.shape[0], cc % tile.shape[1]].copy()
    flecks = rng.random(grid.shape) < .12
    grid[flecks] = rng.choice(palette, np.count_nonzero(flecks))
    group_colors = [int(colors[2]), int(colors[2] if rng.random() < .4 else colors[3])]
    panel_colors = [int(colors[4]), int(colors[4] if rng.random() < .4 else colors[5])]
    for i in range(2):
        gh, top = heights[i], tops[i]
        gw = int(rng.integers(gw_min, width + 1))
        left = int(rng.integers(width - gw + 1))
        _outline(grid, (top, left, top + gh, left + gw), group_colors[i])
        gap = int(rng.integers(1, gw - 2 * pw - 1))
        first = int(rng.integers(1, gw - 2 * pw - gap))
        for slot, along in enumerate((first, first + pw + gap)):
            r = top + int(rng.integers(1, gh - ph))
            c = left + along
            _outline(grid, (r, c, r + ph, c + pw), panel_colors[i])
            grid[r + 1:r + 1 + h, c + 1:c + 1 + w] = contents[i][slot]
    return grid


def _transform(a, turns, reflect):
    result = np.rot90(a, turns)
    if reflect:
        result = result[:, ::-1]
    return result.copy()


def _make_pair(rng, shape, decoys, colors, turns):
    neutral, highlight = map(int, colors[:2])
    pool = colors[6:]
    palettes = [rng.choice(pool, size=int(rng.integers(1, 4)), replace=False) for _ in range(2)]
    source, ref_box = _field(rng, shape, neutral, palettes[0], decoys)
    for _ in range(32):
        target, target_box = _field(rng, shape, neutral, palettes[1], decoys)
        if target_box != ref_box and not np.array_equal(target, source):
            break
    else:
        # This corner construction and its opposite cannot both have the same
        # bounds as the reference, including when that reference is centered.
        spare = _constructive_field(shape, neutral, palettes[1], decoys)
        options = [spare, np.flip(spare)]
        for candidate in options:
            candidate_box = _largest(candidate, neutral)
            if candidate_box != ref_box:
                target, target_box = candidate.copy(), candidate_box
                break
        else:
            raise AssertionError("constructive spare geometry did not move")
    painted = _paint(source, ref_box, highlight)
    reference = [source, painted]
    target_pair = [target, np.full(shape, neutral, dtype=np.int8)]
    if rng.integers(2):
        reference.reverse()
    if rng.integers(2):
        target_pair.reverse()
    contents = [reference, target_pair]
    if rng.integers(2):
        contents.reverse()
    canvas = _layout(rng, shape, contents, colors)
    reflect = bool(rng.integers(2))
    return {"input": _transform(canvas, turns, reflect),
            "output": _transform(_paint(target, target_box, highlight), turns, reflect)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # These preferences only change sampling, never the input-reading rule.
    regime = int(rng.integers(3))  # unseen scale, extra competitors, or orientation
    orientation = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen, first_highlight = set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            if regime == 0:
                shape = (10, int(rng.integers(8, 11))) if test else (
                    int(rng.integers(8 if index == 0 else 6, 10)),
                    int(rng.integers(8 if index == 0 else 6, 10)))
            else:
                minimum = 9 if test and regime == 1 else (8 if index == 0 else 6)
                shape = tuple(int(x) for x in rng.integers(minimum, 11, size=2))
            decoys = 2 if test and regime == 1 else (
                1 if min(shape) >= 8 and (index == 0 or rng.random() < .65) else 0)
            colors = rng.permutation(10)
            if split == "train" and index == 1 and int(colors[1]) == first_highlight:
                colors[[1, 6]] = colors[[6, 1]]
            if split == "train" and index == 0:
                first_highlight = int(colors[1])
            axis = (1 - orientation if test else orientation) if regime == 2 else int(rng.integers(2))
            turns = axis + 2 * int(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, shape, decoys, colors, turns)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
    return puzzle


def _rival_area(panel, neutral, winner):
    rival = panel.copy()
    r, c, b, d = winner
    rival[r:b, c:d] = -1
    return max(map(_area, _bands(rival == neutral)), default=0)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, highlights = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            source, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(source)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _interpret(source)
            neutral, highlight = scene["neutral"], scene["highlight"]
            target, box = scene["target"], scene["target_box"]
            structural = {g[4] for g in scene["groups"]} | {
                f[4] for members in scene["frames"] for f in members}
            group_roles = {g[4] for g in scene["groups"]}
            panel_roles = {f[4] for members in scene["frames"] for f in members}
            _require(not group_roles & panel_roles, "group and local frame roles must differ")
            frame_mask = np.zeros(source.shape, dtype=bool)
            for frame in scene["groups"] + [f for members in scene["frames"] for f in members]:
                r, c, bottom, right = frame[:4]
                frame_mask[r, c:right] = frame_mask[bottom - 1, c:right] = True
                frame_mask[r:bottom, c] = frame_mask[r:bottom, right - 1] = True
            _require(not set(map(int, np.unique(source[~frame_mask]))) & structural,
                     "structural colors must be reserved for the detected outlines")
            _require(out.shape == target.shape, "return exactly the blank panel's interior")
            mask = _interior(target.shape, box)
            _require(np.array_equal(out != target, mask), "changes must equal the strict interior")
            _require(np.all(out[mask] == highlight), "highlight must come from the complete pair")
            _require(np.array_equal(out[~mask], target[~mask]), "preserve perimeter and noise")
            _require(box != scene["ref_box"], "reference copying must not solve generated targets")
            _require(not np.array_equal(out, scene["partner"]), "answer copies the reference")
            for panel, winner in ((scene["source"], scene["ref_box"]), (target, box)):
                _require(_rival_area(panel, neutral, winner) <= .8 * _area(winner), "rival too close in area")
                _require(np.any((panel == neutral) & _ring(panel.shape, winner)), "missing partial neutral contact")
            if split == "train":
                highlights.append(highlight)
    _require(len(set(highlights)) >= 2, "training must disprove a fixed highlight color")
    first = _interpret(puzzle["train"][0]["input"])
    for panel in (first["source"], first["target"]):
        candidates = [b for b in _maximal_boxes(panel, first["neutral"])
                      if b[2] - b[0] >= 3 and b[3] - b[1] >= 3]
        _require(len(candidates) >= 2, "first demonstration needs competing blank windows")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _replace_panel(grid, frame, panel):
    r, c, b, d = frame[:4]
    grid[r + 1:b - 1, c + 1:d - 1] = panel


def _diagnostics():
    puzzle = generate_puzzle(314159, 2, 1)
    canvas = puzzle["train"][0]["input"]
    scene = _interpret(canvas)
    # Full-canvas counterfactual: identical occupied target and all other cells,
    # except the demonstrated highlight in the complete reference partner.
    source = np.full((8, 8), 6, dtype=np.int8)
    source[0:4, 0:4] = 0
    source[5:8, 5:8] = 0
    source[4, 0] = 0  # partial side contact, not a larger rectangle
    target = source[::-1].copy()
    contents = [[source, _paint(source, (0, 0, 4, 4), 1)],
                [target, np.zeros((8, 8), dtype=np.int8)]]
    local = _layout(np.random.default_rng(410), (8, 8), contents, np.arange(10))
    local[local == 7] = 6  # leave the alternative highlight absent everywhere
    local_scene = _interpret(local)
    twin = local.copy()
    partner_frame = local_scene["frames"][local_scene["reference_group"]][1 - local_scene["source_slot"]]
    _replace_panel(twin, partner_frame, _paint(source, (0, 0, 4, 4), 7))
    first, second = solve(local), solve(twin)
    _require(not np.array_equal(first, second), "reference highlight has no effect")
    target_box = _largest(target, 0)
    _require(np.array_equal(first, _paint(target, target_box, 1)), "bad original answer")
    _require(np.array_equal(second, _paint(target, target_box, 7)), "bad twin answer")
    # A's neutral-role evidence also matters: keep both raw panels unchanged,
    # but demonstrate the operation on a different neutral color. The smaller
    # monochrome block then becomes eligible, despite a larger other-color one.
    source = np.full((8, 8), 8, dtype=np.int8)
    source[0:4, 0:4] = 0
    source[5:8, 4:8] = 6
    target = source[::-1].copy()
    role_answers = []
    for neutral in (0, 6):
        ref_box = _largest(source, neutral)
        contents = [[source, _paint(source, ref_box, 1)],
                    [target, np.full((8, 8), neutral, dtype=np.int8)]]
        # Identical layout/wallpaper and raw panels in both counterfactuals.
        local = _layout(np.random.default_rng(411), (8, 8), contents, np.arange(10))
        answer = solve(local)
        _require(np.array_equal(answer, _paint(target, _largest(target, neutral), 1)),
                 "blank/reference evidence did not determine the neutral role")
        role_answers.append(answer)
    _require(not np.array_equal(*role_answers), "neutral-role counterfactual failed")
    # Permuting the reference members, or target slots, cannot change the answer.
    for group in (scene["reference_group"], scene["target_group"]):
        twin = canvas.copy()
        f0, f1 = scene["frames"][group]
        a, b = _crop(canvas, f0), _crop(canvas, f1)
        _replace_panel(twin, f0, b)
        _replace_panel(twin, f1, a)
        _require(np.array_equal(solve(twin), scene["answer"]), "slot position selected the operation")
    for turns in range(4):
        for reflect in (False, True):
            _require(np.array_equal(solve(_transform(canvas, turns, reflect)),
                                    _transform(scene["answer"], turns, reflect)),
                     "whole-scene orientation changed the rule")
    for h in range(6, 11):
        for w in range(6, 11):
            for decoys in range(3 if min(h, w) >= 8 else 1):
                spare = _constructive_field((h, w), 0, [6, 8], decoys)
                winner = _largest(spare, 0)
                _require(_rival_area(spare, 0, winner) <= .8 * _area(winner),
                         "constructive fallback has an oversized rival")
                _require(np.any((spare == 0) & _ring(spare.shape, winner)),
                         "constructive fallback lacks a partial contact")
    # Area beats longest side; both rectangles are inclusion-maximal.
    panel = np.full((10, 10), 6, dtype=np.int8)
    panel[0:5, 0:5] = 0
    panel[7:10, 0:6] = 0
    box = _largest(panel, 0)
    _require(box == (0, 0, 5, 5), "selection did not use global area")
    out = _paint(panel, box, 1)
    _require(out[0, 0] == out[4, 4] == 0 and np.all(out[1:4, 1:4] == 1), "strict interior failed at boundary")
    panel[7:10, 0:6] = 6
    panel[7:10, 0:3] = 0
    panel[0:5, 0:5] = 6
    panel[0:3, 0:3] = 0
    try:
        _largest(panel, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("equal greatest areas must be rejected")


def self_test():
    """42 seeds, streamed; no saved grids, traces, logs, or unbounded retries."""
    metrics = {k: set() for k in ("shapes", "panel_shapes", "rectangles", "roles", "positions",
                                  "rivals", "orders", "palettes", "counts", "structures")}
    pairs, boundary, squares, nonsquares, longer_rivals = 0, 0, 0, 0, 0
    for seed in range(42):
        # Count range is exercised without multiplying all seeds by all counts.
        nt = 1 + seed % 4
        nr = max(nt + 1, 2 + (seed * 3) % 9)
        puzzle = generate_puzzle(seed, nr, nt)
        validate_generated_puzzle(puzzle)
        _require(_equal(puzzle, generate_puzzle(seed, nr, nt)), "nondeterministic generation")
        metrics["counts"].add((nr, nt))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _interpret(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve differs")
                target, box = s["target"], s["target_box"]
                r, c, b, d = box
                metrics["shapes"].add(pair["input"].shape)
                metrics["panel_shapes"].add(target.shape)
                metrics["rectangles"].add((b - r, d - c))
                metrics["roles"].add((s["neutral"], s["highlight"]))
                metrics["positions"].add((r < (target.shape[0] - b + r) / 2,
                                          c < (target.shape[1] - d + c) / 2))
                metrics["orders"].add((s["reference_group"], s["source_slot"], s["blank_slot"]))
                metrics["rivals"].add(_rival_area(target, s["neutral"], box))
                metrics["palettes"].add(tuple(map(int, np.unique(target))))
                # Hashes avoid retaining duplicate reconstructible payloads.
                normalized, roles = [], {}
                for x in target.flat:
                    normalized.append(roles.setdefault(int(x), len(roles)))
                metrics["structures"].add(hashlib.sha256(bytes(normalized)).digest())
                boundary += r == 0 or c == 0 or b == target.shape[0] or d == target.shape[1]
                squares += b - r == d - c
                nonsquares += b - r != d - c
                for rival in _maximal_boxes(target, s["neutral"]):
                    if _disjoint(rival, box) and max(rival[2] - rival[0], rival[3] - rival[1]) > max(b - r, d - c):
                        longer_rivals += 1
                pairs += 1
    minima = {"shapes": 25, "panel_shapes": 12, "rectangles": 15, "roles": 30,
              "positions": 4, "rivals": 8, "orders": 8, "palettes": 40, "counts": 10, "structures": 150}
    for name, minimum in minima.items():
        _require(len(metrics[name]) >= minimum, f"insufficient {name} variation")
    _require(min(boundary, squares, nonsquares, longer_rivals) > 0, "missing structural extremes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1200 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (None, 1), (4, None), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(90, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a wrong output")
    _diagnostics()
    return {"seeds": 42, "pairs": pairs, "distinct": {k: len(v) for k, v in metrics.items()},
            "boundary_winners": boundary, "square_winners": squares,
            "nonsquare_winners": nonsquares, "longer_side_rivals": longer_rivals}
