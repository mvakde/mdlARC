"""Topology-selective finite contact probes; standard library and NumPy only."""
from __future__ import annotations

import itertools
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells, directions=D4):
    unseen = set(cells)
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component, todo = {first}, [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _holes(cells):
    """Enclosed four-connected complement regions, with genuine exterior padding."""
    r0, r1, c0, c1 = _bbox(cells)
    complement = {(r, c) for r in range(r0 - 1, r1 + 2)
                  for c in range(c0 - 1, c1 + 2)} - set(cells)
    return [region for region in _components(complement)
            if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in region)]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _local(p, endpoint, forward, lateral):
    dr, dc = p[0] - endpoint[0], p[1] - endpoint[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _global(p, endpoint, forward, lateral):
    u, v = p
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _cap(cells, length):
    peak = max(v for u, v in cells)
    outgoing = max(u + v for u, v in cells)
    return [(u, min(u + 1, peak + 1, outgoing + 1 - u)) for u in range(length)]


def _touches(cells, route):
    return any((r + dr, c + dc) in route for r, c in cells for dr, dc in D4)


def _as_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _parse(input_grid):
    grid = _as_grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(counts.argmax())
    assert np.count_nonzero(counts == counts[background]) == 1, "background frequency tie"
    colors = [int(c) for c in np.unique(grid) if c != background]
    assert len(colors) == 3, "three foreground roles required"
    color_cells = {c: {tuple(map(int, p)) for p in np.argwhere(grid == c)} for c in colors}
    components4 = {c: _components(color_cells[c]) for c in colors}
    components8 = {c: _components(color_cells[c], D8) for c in colors}
    candidates = []
    # The post color has multiple spatially separate components; the two other
    # colors are the single cuff/stem and the single eight-connected strand.
    for post_color in colors:
        posts = components4[post_color]
        if not 2 <= len(posts) <= 5 or len(components8[post_color]) != len(posts):
            continue
        for anchor_color in colors:
            if anchor_color == post_color or len(components4[anchor_color]) != 1:
                continue
            strand_color = next(c for c in colors if c not in (post_color, anchor_color))
            if len(components8[strand_color]) != 1:
                continue
            anchor, strand = color_cells[anchor_color], color_cells[strand_color]
            if len(strand) < 2:
                continue
            key_holes = _holes(anchor)
            key = len(key_holes)
            if not 0 <= key <= 3:
                continue
            hole_lists = [_holes(post) for post in posts]
            if any(len(h) > 4 for h in hole_lists):
                continue
            if any(grid[p] != background for holes in [key_holes] + hole_lists
                   for cavity in holes for p in cavity):
                continue
            for endpoint in sorted(anchor):
                for forward in D4:
                    if any((endpoint[0] - j * forward[0], endpoint[1] - j * forward[1])
                           not in anchor for j in (1, 2)):
                        continue
                    if (endpoint[0] + forward[0], endpoint[1] + forward[1]) in anchor:
                        continue
                    for lateral in ((forward[1], -forward[0]), (-forward[1], forward[0])):
                        if _global((0, 1), endpoint, forward, lateral) not in strand:
                            continue
                        a = {_local(p, endpoint, forward, lateral) for p in anchor}
                        t = {_local(p, endpoint, forward, lateral) for p in strand}
                        if (any(u > 0 or (u >= -2 and v != 0) for u, v in a)
                                or not any(v < 0 for u, v in a)
                                or not any(v > 0 for u, v in a)
                                or any(u > 0 or v < 1 or (u == 0 and v != 1) for u, v in t)):
                            continue
                        local_posts = [{_local(p, endpoint, forward, lateral) for p in post}
                                       for post in posts]
                        if any(u < 2 for post in local_posts for u, v in post):
                            continue
                        eligible = [i for i, holes in enumerate(hole_lists) if len(holes) == key]
                        if not eligible:
                            continue
                        selected = set().union(*(local_posts[i] for i in eligible))
                        peak = max(v for u, v in selected)
                        if (peak < 0 or any(v > u for u, v in selected)
                                or (peak != 0 and max(v - u for u, v in selected) != 0)):
                            continue
                        path = [_global(p, endpoint, forward, lateral)
                                for p in _cap(selected, len(strand))]
                        visible = {p for p in path if _inside(p, grid.shape)}
                        if visible & (anchor | set().union(*(posts[i] for i in eligible))):
                            continue
                        contacted = [i for i in eligible if _touches(posts[i], visible)]
                        candidates.append(dict(grid=grid, background=background,
                                               anchor_color=anchor_color, strand_color=strand_color,
                                               post_color=post_color, anchor=anchor, strand=strand,
                                               key=key, posts=posts, hole_lists=hole_lists,
                                               local_posts=local_posts, eligible=eligible,
                                               selected=selected, endpoint=endpoint, forward=forward,
                                               lateral=lateral, path=path, visible=visible,
                                               contacted=contacted))
    assert len(candidates) == 1, "input must have one geometric/topological interpretation"
    return candidates[0]


def _answer(parsed):
    out = parsed["grid"].copy()
    for p in parsed["strand"] | set().union(*parsed["posts"]):
        out[p] = parsed["background"]
    for i in parsed["contacted"]:
        for p in parsed["posts"][i]:
            out[p] = parsed["anchor_color"]
    for p in parsed["visible"]:
        out[p] = parsed["strand_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions, geometry, topology and length come from input pixels."""
    return _answer(_parse(input_grid))


def _mask_cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _valid_mask(mask, holes):
    cells = _mask_cells(mask)
    return bool(cells) and len(_components(cells)) == 1 and len(_holes(cells)) == holes


def _object_mask(k, rng, simple=False, far_corner=False):
    # Compact lattices are only a starting topology, never a stored scene.
    rows = 2 if k >= 3 and not simple and rng.random() < .35 else 1
    cols = max(1, (k + rows - 1) // rows)
    mask = np.ones((2 * rows + 1, max(3, 2 * cols + 1)), dtype=bool)
    for i in range(k):
        mask[1 + 2 * (i // cols), 1 + 2 * (i % cols)] = False
    if not simple:
        if k == 0 and rng.random() < .45:
            # Open C/U objects have no cavity despite a visibly large bay.
            mask = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
        if rng.random() < .40:
            axis = int(rng.integers(2))
            index = int(rng.integers(mask.shape[axis]))
            mask = np.insert(mask, index, np.take(mask, index, axis=axis), axis=axis)
        # Enlarge individual cavities and carve exterior notches; recompute
        # actual topology so neither foreground islands nor accidental holes
        # are accepted. Edits can produce unequal/nonrectangular cavities.
        for _ in range(10):
            candidate = mask.copy()
            r, c = (int(rng.integers(d)) for d in mask.shape)
            candidate[r, c] = False
            if _valid_mask(candidate, k):
                mask = candidate
        mask = np.rot90(mask, int(rng.integers(4))).copy()
    if far_corner:
        # This pixel matters geometrically even though a diagonal opening does
        # not change the four-connected cavity class.
        candidate = mask.copy()
        candidate[-1, -1] = False
        if _valid_mask(candidate, k) and candidate[-1].any() and candidate[:, -1].any():
            mask = candidate
        else:
            mask = np.ones((3, max(3, 2 * k + 1)), dtype=bool)
            for i in range(k):
                mask[1, 1 + 2 * i] = False
            mask[-1, -1] = False
    cells = _mask_cells(mask)
    r0, r1, c0, c1 = _bbox(cells)
    return mask[r0:r1 + 1, c0:c1 + 1].copy()


def _anchor(k, stem, rng, simple):
    height = 3 if simple else int(rng.integers(3, 5))
    width = max(3, 2 * k + 1) + (0 if simple else 2 * int(rng.integers(2)))
    mask = np.ones((height, width), dtype=bool)
    for i in range(k):
        mask[1, 1 + 2 * i] = False
    # Widen selected holes while preserving a single connected prototype.
    if not simple and height == 4 and rng.random() < .6:
        for i in range(k):
            if rng.random() < .5:
                mask[2, 1 + 2 * i] = False
    assert _valid_mask(mask, k)
    cuff = {(r - stem - height + 1, c - width // 2) for r, c in _mask_cells(mask)}
    return cuff | {(-u, 0) for u in range(stem)}


def _slack(length, stem, rng, diagonal, simple):
    depth = stem - 2
    width = max(3, (length - 1 + depth - 1) // depth + int(diagonal))
    width += 0 if simple else int(rng.integers(0, 3))
    low_v = 2 if diagonal else 1
    cells = {(0, 1), (-1, low_v)}
    style = 0 if simple else int(rng.integers(3))
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in D8
                           if -depth <= u + du <= -1 and low_v <= v + dv <= width} - cells)
        assert frontier
        weights = []
        for u, v in frontier:
            neighbors = sum((u + du, v + dv) in cells for du, dv in D8)
            weights.append((neighbors + 1) ** 2 if style == 0 else
                           1.0 / (neighbors + 1) ** 2 if style == 1 else 1.0)
        weights = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _place_post(mask, u, top):
    cells = _mask_cells(mask)
    return {(r + u, c + top - (mask.shape[1] - 1)) for r, c in cells}


def _construct_pair(rng, roles, key, outer_count, lower, wrong, full,
                    clip, axis, lesson=None, simple=False):
    bg, ac, tc, oc = roles
    stem = 4 if simple else int(rng.integers(4, 7))
    anchor = _anchor(key, stem, rng, simple)
    peak = 3 if simple else int(rng.choice([0, 3, 4]))
    posts, classes = [], []
    for i in range(outer_count):
        mask = _object_mask(key, rng, simple, far_corner=(lesson == "long" and i == outer_count - 1))
        if i == 0:
            shifted = _place_post(mask, 0, peak)
            near = int(rng.integers(2, 5)) if peak == 0 else max(v - u for u, v in shifted)
            near = max(2, near)
            # A rare heavily sculpted mask can have its top too far behind its
            # front for the 45-degree tangent; choose a larger visible height.
            if peak != 0 and max(v - u for u, v in shifted) < 2:
                peak += 2 - max(v - u for u, v in shifted)
                shifted = _place_post(mask, 0, peak)
                near = max(v - u for u, v in shifted)
        else:
            near = max(u for u, v in posts[-1]) + (2 if simple else int(rng.integers(2, 5)))
        posts.append(_place_post(mask, near, peak))
        classes.append(key)
    if lower:
        mask = _object_mask(key, rng, simple)
        top = min(v for u, v in posts[0]) - (3 if simple else int(rng.integers(3, 5)))
        near = min(u for u, v in posts[0]) + int(rng.integers(0, 2))
        posts.append(_place_post(mask, near, top))
        classes.append(key)
    if wrong:
        other = int(rng.choice([k for k in range(4) if k != key]))
        mask = _object_mask(other, rng, simple)
        bottom = peak + 3
        top = bottom + mask.shape[1] - 1
        # A tall wrong-class post would itself admit a diagonal tangent if it
        # were solid; ignoring topology is therefore a meaningful ablation.
        near = max(2, max(v - u for u, v in _place_post(mask, 0, top)))
        posts.append(_place_post(mask, near, top))
        classes.append(other)
    selected = set().union(*(p for p, k in zip(posts, classes) if k == key))
    far = max(u for p in posts[:outer_count] for u, v in p)
    if full:
        length = far + (3 if simple else int(rng.integers(2, 6)))
    else:
        # At least one contact with the first post, but no contact with the
        # second outer post. Ending immediately after its first supported cell
        # makes the length lesson visible even for the two-example minimum.
        prospective = set(_cap(selected, far + 2))
        first_contacts = [u for u, v in prospective
                          if _touches(posts[0], {(u, v)})]
        first_u = min(first_contacts)
        second_u = min(u for u, v in prospective
                      if _touches(posts[1], {(u, v)})) if outer_count > 1 else far + 3
        length = int(rng.integers(first_u + 2, max(first_u + 3, second_u + 1)))
        length = min(length, second_u)
    input_forward_max = max(u for p in posts for u, v in p)
    if clip:
        length = max(length, input_forward_max + 3)
    strand = _slack(length, stem, rng, diagonal=(lesson == "short" or rng.random() < .55), simple=simple)
    path = _cap(selected, length)
    input_cells = anchor | strand | set().union(*posts)
    all_cells = input_cells | set(path)
    u0, u1, v0, v1 = _bbox(all_cells)
    if clip:
        u1 = input_forward_max
    margins = [0, 0, 0, 0] if simple else list(map(int, rng.integers(0, 3, size=4)))
    if clip:
        margins[1] = 0
    u0 -= margins[0]
    u1 += margins[1]
    v0 -= margins[2]
    v1 += margins[3]
    if max(u1 - u0 + 1, v1 - v0 + 1) > 30:
        return None
    inp = np.full((u1 - u0 + 1, v1 - v0 + 1), bg, dtype=np.int8)
    def position(p):
        return p[0] - u0, p[1] - v0
    for cells, color in [(anchor, ac), (strand, tc)] + [(post, oc) for post in posts]:
        for p in cells:
            q = position(p)
            if inp[q] != bg:
                return None
            inp[q] = color
    # Construction uses its own known cells, not the inferred role assignment.
    visible_local = {p for p in path if _inside(position(p), inp.shape)}
    out = inp.copy()
    for p in strand | set().union(*posts):
        out[position(p)] = bg
    for post, k in zip(posts, classes):
        if k == key and _touches(post, visible_local):
            for p in post:
                out[position(p)] = ac
    for p in visible_local:
        out[position(p)] = tc
    # The local frame may independently reflect its lateral axis, and the
    # puzzle-level teaching regime controls only the preferred forward axis.
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    rotation = (0 if axis == 0 else 1) + 2 * int(rng.integers(2))
    pair = {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    try:
        parsed = _parse(pair["input"])
        assert np.array_equal(_answer(parsed), pair["output"])
        if lesson == "long":
            assert len(parsed["contacted"]) >= 2
            assert any(i not in parsed["contacted"] for i in parsed["eligible"])
            assert _bridge_witness(parsed)
            assert _bbox_witness(parsed)
            assert _topology_witness(parsed)
        elif lesson == "short":
            assert len(parsed["contacted"]) == 1 and len(parsed["eligible"]) >= 2
            assert len(parsed["visible"]) == len(parsed["strand"])
            assert len(_components(parsed["strand"])) > 1
        assert not np.array_equal(pair["input"], pair["output"])
    except AssertionError:
        return None
    return pair


def _topology_witness(parsed):
    all_posts = set().union(*parsed["local_posts"])
    other = {_global(p, parsed["endpoint"], parsed["forward"], parsed["lateral"])
             for p in _cap(all_posts, len(parsed["strand"]))}
    return {p for p in other if _inside(p, parsed["grid"].shape)} != parsed["visible"]


def _bbox_witness(parsed):
    boxes = set()
    for i in parsed["eligible"]:
        u0, u1, v0, v1 = _bbox(parsed["local_posts"][i])
        boxes.update((u, v) for u in range(u0, u1 + 1) for v in range(v0, v1 + 1))
    other = {_global(p, parsed["endpoint"], parsed["forward"], parsed["lateral"])
             for p in _cap(boxes, len(parsed["strand"]))}
    return {p for p in other if _inside(p, parsed["grid"].shape)} != parsed["visible"]


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(4))
    base_axis = int(rng.integers(2))
    clip_lesson = bool(rng.random() < .40)
    teaching_keys = list(map(int, rng.permutation(2 if regime == 2 else 4)[:2]))
    puzzle = {"train": [], "test": []}
    used_roles = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            # Distinct role tuples ensure distinct inputs because input parsing
            # is unique. There are 5,040 tuples and at most fourteen pairs.
            for _ in range(20):
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
                if roles not in used_roles:
                    break
            else:
                roles = next(p for p in itertools.permutations(range(10), 4) if p not in used_roles)
            used_roles.add(roles)
            lesson = "long" if split == "train" and index == 0 else "short" if split == "train" and index == 1 else None
            if lesson:
                key = teaching_keys[index]
                outer, lower, wrong, full = (2, True, True, True) if lesson == "long" else (2, False, True, False)
                clip = clip_lesson if lesson == "long" else False
            else:
                if regime == 2:
                    key = int(rng.integers(2, 4)) if split == "test" else int(rng.integers(2))
                else:
                    key = int(rng.integers(4))
                outer = (3 if split == "test" and regime == 1 else
                         int(rng.integers(1, 3 if regime == 1 else 4)))
                lower, wrong = bool(rng.random() < .55), bool(rng.random() < .85)
                if outer + lower + wrong < 2:
                    wrong = True
                full = bool(rng.random() < .55) or outer == 1
                if split == "test" and regime == 1:
                    full = True
                if regime == 3:
                    if split == "test":
                        # Recombine the separately taught recessed match and
                        # finite reach, with a short strand in a four-post scene.
                        outer, lower, wrong, full = 2, True, True, False
                    elif not full:
                        lower = False
                        if outer == 1 and not wrong:
                            wrong = True
                clip = clip_lesson and full and rng.random() < .40
            axis = 1 - base_axis if split == "test" and regime == 0 else base_axis
            if regime == 3:
                axis = int(rng.integers(2))
            for attempt in range(48):
                pair = _construct_pair(rng, roles, key, outer, lower, wrong, full,
                                       clip, axis, lesson, simple=attempt >= 32)
                if pair is not None:
                    break
            else:
                # Guaranteed small layout keeps the same key, lesson, axis and
                # clipping convention. Optional scenes use the long/short
                # construction too, rather than a separate answer rule.
                pair = _construct_pair(rng, roles, key, 2, lesson != "short", True,
                                       lesson != "short", clip, axis, lesson, simple=True)
                assert pair is not None, "bounded fitting construction failed"
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _bridge_witness(parsed):
    if len(parsed["contacted"]) < 2:
        return False
    # A visible level route cell with no neighboring eligible material lies
    # strictly between two contacted objects: the strand bridges air.
    route = [_local(p, parsed["endpoint"], parsed["forward"], parsed["lateral"])
             for p in parsed["path"] if p in parsed["visible"]]
    contact_us = [u for u, v in route if _touches(parsed["selected"], {(u, v)})]
    for a, b in zip(route, route[1:]):
        if (a[1] == b[1] and min(contact_us) < a[0] < max(contact_us)
                and not _touches(parsed["selected"], {a})):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_as_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = _signature(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            parsed = _parse(inp)
            assert np.array_equal(_answer(parsed), out), "wrong contact-probe answer"
            assert len(parsed["visible"]) <= len(parsed["strand"])
            assert _global((0, 1), parsed["endpoint"], parsed["forward"], parsed["lateral"]) in parsed["visible"]
            assert all(out[p] == parsed["anchor_color"] for p in parsed["anchor"])
            for i, post in enumerate(parsed["posts"]):
                if i in parsed["contacted"]:
                    assert all(out[p] == parsed["anchor_color"] for p in post)
                else:
                    assert all(out[p] == (parsed["strand_color"] if p in parsed["visible"]
                                          else parsed["background"]) for p in post)
            expected_changes = parsed["strand"] | set().union(*parsed["posts"]) | parsed["visible"]
            assert all(out[r, c] == inp[r, c] for r in range(inp.shape[0])
                       for c in range(inp.shape[1]) if (r, c) not in expected_changes)
            if split == "train":
                training.append(parsed)
    assert len({p["key"] for p in training}) >= 2, "missing changed prototype class"
    assert any(_topology_witness(p) for p in training), "no geometry-changing wrong-class post"
    assert any(len(p["contacted"]) >= 2 and _bridge_witness(p) for p in training), "missing air bridge"
    assert any(_bbox_witness(p) for p in training), "no actual-contour departure witness"
    assert any(i not in p["contacted"] for p in training for i in p["eligible"]), "missing untouched match"
    assert any(len(p["visible"]) == len(p["strand"]) and len(_components(p["strand"])) > 1
               for p in training), "missing eight-connected length demonstration"
    assert any(len(p["contacted"]) == 1 and len(p["eligible"]) >= 2 for p in training), "missing partial reach"


def _features(pair):
    p = _parse(pair["input"])
    return dict(shape=tuple(p["grid"].shape), roles=(p["background"], p["anchor_color"], p["strand_color"], p["post_color"]),
                forward=p["forward"], key=p["key"], posts=len(p["posts"]), eligible=len(p["eligible"]),
                touched=len(p["contacted"]), length=len(p["strand"]), clipped=len(p["visible"]) < len(p["strand"]),
                peak=max(v for u, v in p["selected"]),
                cavity_areas=tuple(len(h) for holes in p["hole_lists"] for h in holes),
                silhouettes=tuple(tuple(sorted((u - _bbox(cells)[0], v - _bbox(cells)[2]) for u, v in cells))
                                  for cells in p["local_posts"]),
                irregular_cavities=sum(len(h) < ((_bbox(h)[1] - _bbox(h)[0] + 1) *
                                                (_bbox(h)[3] - _bbox(h)[2] + 1)) for holes in p["hole_lists"] for h in holes),
                bbox_witness=_bbox_witness(p), topology_witness=_topology_witness(p))


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _contact_regressions():
    """Hand-specified geometry and counterfactuals; no generated labels involved."""
    origin = (7, 7)
    def at(p):
        return p[0] + origin[0], p[1] + origin[1]
    cuff = {(u, v) for u in (-6, -5, -4) for v in (-1, 0, 1)} - {(-5, 0)}
    anchor = cuff | {(u, 0) for u in (-3, -2, -1, 0)}
    first = {(u, v) for u in (3, 4, 5) for v in (1, 2, 3)} - {(4, 2)}
    last = {(u, v) for u in (8, 9, 10) for v in (1, 2, 3)} - {(9, 2), (10, 3)}
    recessed = {(u, v) for u in (4, 5, 6) for v in (-6, -5, -4)} - {(5, -5)}
    wrong = {(u, v) for u in (8, 9, 10) for v in (6, 7, 8)}
    strand_order = [(0, 1)] + [(-1, v) for v in range(2, 9)] + [(-2, v) for v in range(2, 8)]
    expected_route = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4), (6, 4),
                      (7, 4), (8, 4), (9, 4), (10, 3), (11, 2), (12, 1), (13, 0)]
    assert len(strand_order) == 14
    for length in (8, 14):
        inp = np.zeros((22, 19), dtype=np.int8)
        for p in anchor:
            inp[at(p)] = 4
        for post in (first, last, recessed, wrong):
            for p in post:
                inp[at(p)] = 5
        for p in strand_order[:length]:
            inp[at(p)] = 2
        expected = np.zeros_like(inp)
        for p in anchor | first | (last if length == 14 else set()):
            expected[at(p)] = 4
        for p in expected_route[:length]:
            expected[at(p)] = 2
        assert np.array_equal(solve(inp), expected)
        parsed = _parse(inp)
        assert len(parsed["contacted"]) == (2 if length == 14 else 1)
        assert _topology_witness(parsed)
        if length == 14:
            assert _bbox_witness(parsed) and _bridge_witness(parsed)
            # Filling one cuff cavity changes eligibility from rings to the
            # solid tall post, and consequently changes both route and survivors.
            altered = inp.copy()
            altered[at((-5, 0))] = 4
            alternate = np.zeros_like(inp)
            for p in anchor | {(-5, 0)} | wrong:
                alternate[at(p)] = 4
            high_route = [(u, u + 1) for u in range(9)] + [(9, 9), (10, 9), (11, 8), (12, 7), (13, 6)]
            for p in high_route:
                alternate[at(p)] = 2
            assert np.array_equal(solve(altered), alternate)
            assert not np.array_equal(alternate, expected)
            # Rearrange the slack without changing its length or attachment.
            rearranged = inp.copy()
            for p in strand_order:
                rearranged[at(p)] = 0
            replacement = {(0, 1)} | {(-u, v) for u in (1, 2) for v in range(1, 8)}
            replacement.remove((-2, 7))
            assert len(replacement) == 14
            for p in replacement:
                rearranged[at(p)] = 2
            assert np.array_equal(solve(rearranged), expected)


def self_test():
    """Bounded pixel invariants, counterfactuals, and 42-seed structural audit."""
    diagonal = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert len(_holes(diagonal)) == 1
    assert len(_holes(diagonal - {(0, 1)})) == 0
    assert len(_holes({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)})) == 0
    assert _cap({(3, 3), (4, 3), (7, 3), (8, 2)}, 11) == [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 3), (9, 2), (10, 1)]
    _contact_regressions()
    features, structures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(seed))
        local_features = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                # Color permutation and all rotations/reflections preserve the
                # input-only rule, even when they alter the teaching regime.
                permutation = np.array([(i + 3) % 10 for i in range(10)], dtype=np.int8)
                transformed = np.rot90(permutation[inp][:, ::-1], seed % 4).copy()
                expected = np.rot90(permutation[out][:, ::-1], seed % 4).copy()
                assert np.array_equal(solve(transformed), expected)
                f = _features(pair)
                local_features.append(f)
                structures.add((f["shape"], f["forward"], f["key"], f["length"], f["silhouettes"]))
        assert len({f["roles"] for f in local_features}) == len(local_features)
        features.extend(local_features)
    assert len({f["shape"] for f in features}) >= 30
    assert len(structures) >= 180
    assert {f["forward"] for f in features} == set(D4)
    assert {f["key"] for f in features} == {0, 1, 2, 3}
    assert {f["posts"] for f in features} >= {2, 3, 4, 5}
    assert {f["touched"] for f in features} >= {1, 2, 3}
    assert any(f["clipped"] for f in features) and any(not f["clipped"] for f in features)
    assert any(f["peak"] == 0 for f in features) and any(f["peak"] > 0 for f in features)
    assert len({s for f in features for s in f["silhouettes"]}) >= 30
    assert any(f["irregular_cavities"] for f in features)
    assert max(a for f in features for a in f["cavity_areas"]) >= 3
    assert len({f["length"] for f in features}) >= 10
    assert {f["roles"][0] for f in features} == set(range(10))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(137 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Directly exercise the small bounded construction, including every cuff
    # class, each lesson, both axes and clipped/unclipped long contact.
    rng = np.random.default_rng(707)
    for key in range(4):
        for axis in range(2):
            for lesson, clip in (("long", False), ("long", True), ("short", False)):
                p = _construct_pair(rng, (0, 1, 2, 3), key, 2, lesson == "long", True,
                                    lesson == "long", clip, axis, lesson, simple=True)
                assert p is not None
    damaged = generate_puzzle(17, 2, 1)
    pair = damaged["test"][0]
    parsed = _parse(pair["input"])
    cell = min(parsed["anchor"])
    pair["output"][cell] = parsed["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged preserved anchor accepted")
    return {"seeds": 42, "pairs": len(features), "shapes": len({f["shape"] for f in features}),
            "structures": len(structures), "post_silhouettes": len({s for f in features for s in f["silhouettes"]}),
            "lengths": sorted({f["length"] for f in features}),
            "post_counts": sorted({f["posts"] for f in features}),
            "contact_counts": sorted({f["touched"] for f in features}),
            "clipped_pairs": sum(f["clipped"] for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
