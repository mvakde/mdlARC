"""Reciprocal-compass mosaics: neighbors supply each other's reference frames.

Only the standard library and NumPy are used. The solver searches the current
input, including all dihedral poses, and rejects distinct tied pixel answers.
"""
from __future__ import annotations

import copy
import math
import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = [first]
        for r, c in queue:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(queue)
    return result


def _pose(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _poses(a):
    result, seen = [], set()
    for t in range(8):
        b = _pose(a, t)
        key = (b.shape, b.tobytes())
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


def _point(shape, corner):
    return (shape[0]-1 if corner[0] == "B" else 0,
            shape[1]-1 if corner[1] == "R" else 0)


def _marker(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r+dr, c), (r, c+dc), (r+dr, c+dc))


def _parse(input_grid):
    g = _grid(input_grid)
    hist = np.bincount(g.ravel(), minlength=10)
    bg = int(hist.argmax())
    if np.count_nonzero(hist == hist[bg]) != 1:
        raise ValueError("background must be the uniquely most frequent color")
    markers = []
    for color in range(10):
        if color == bg or hist[color] != 3:
            continue
        q = np.argwhere(g == color)
        lo, hi = q.min(0), q.max(0)
        if tuple(hi-lo) != (1, 1):
            continue
        square = {(int(r), int(c)) for r in (lo[0], hi[0])
                  for c in (lo[1], hi[1])}
        missing = square - set(map(tuple, q.tolist()))
        p = missing.pop()
        if g[p] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL",
                  (0, 0): "BR"}[(p[0]-lo[0], p[1]-lo[1])]
        markers.append((color, p, corner))
    if len(markers) != 1:
        raise ValueError("expected one external three-cell corner marker")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        h, w = map(int, q.max(0)-q.min(0)+1)
        if min(h, w) < 4 or len(cells) != h*w:
            raise ValueError("tiles must be separated filled rectangles, sides >=4")
        a = g[r:r+h, c:c+w].copy()
        identity = int(a[1, 1])
        if not np.all(a[1:-1, 1:-1] == identity):
            raise ValueError("each tile needs a uniform identity-colored interior")
        if point in cells:
            anchor = len(pieces)
            local = _point(a.shape, corner)
            if point != (r+local[0], c+local[1]):
                raise ValueError("marker does not pin a tile corner")
        pieces.append({"array": a, "old": (r, c), "identity": identity})
    if anchor is None or not 3 <= len(pieces) <= 6:
        raise ValueError("expected 3..6 tiles including the marked tile")
    corners = {int(p["array"][r, c]) for p in pieces
               for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
    if len(corners) != 1:
        raise ValueError("all tile corners must share the wall color")
    wall = corners.pop()
    identities = {p["identity"] for p in pieces}
    if wall in identities or bg in identities or marker in identities:
        raise ValueError("color roles must be locally distinct")
    extra = set(map(int, np.unique(g))) - identities - {bg, marker, wall}
    if len(extra) != 1 or len({bg, marker, wall}) != 3:
        raise ValueError("expected one separate registration color")
    registration = extra.pop()
    for p in pieces:
        a = p["array"]
        edge = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
        if not np.any(edge == registration) or not np.any(np.isin(edge, list(identities))):
            raise ValueError("each tile must have registration marks and compass ports")
    return {"grid": g, "background": bg, "marker": marker, "point": point,
            "corner": corner, "wall": wall, "registration": registration,
            "identities": identities, "pieces": pieces, "anchor": anchor}


def _join_ok(first, second, first_id, second_id, info,
             use_registration=True, use_compass=True):
    """Compare one COMPLETE straight contact, in common world-coordinate order.

    Registration ablation retains directional neighbor identities but ignores
    the positions of marks and ports. Compass ablation treats every ID as the
    same port token while retaining exact mark/port positions.
    """
    wall, reg = info["wall"], info["registration"]
    ap = (first != wall) & (first != reg)
    bp = (second != wall) & (second != reg)
    if not (np.any(ap) and np.any(bp)):
        return False
    if use_registration:
        if not (np.any(first == reg) and np.any(second == reg)):
            return False
        if not (np.array_equal(first == wall, second == wall)
                and np.array_equal(first == reg, second == reg)
                and np.array_equal(ap, bp)):
            return False
    if use_compass:
        if np.any(first[ap] != second_id) or np.any(second[bp] != first_id):
            return False
    return True


def _placements(info, use_registration=True, use_compass=True, limit=2,
                identity_only=False):
    """Exhaustive rectangle/pose search; normally distinct final pixels count.

    Filling the first uncovered cell is complete for rectangular tiles. No
    scattered position except the anchor is used as a placement clue.
    identity_only is used solely to probe core-placement ambiguity in ablations.
    """
    pieces = info["pieces"]
    anchor = info["anchor"]
    area = sum(p["array"].size for p in pieces)
    variants = [_poses(p["array"]) for p in pieces]
    solutions, nodes = {}, 0
    wall = info["wall"]
    canvas = info["grid"].shape

    for h in range(4, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        if not 4 <= w <= canvas[1]:
            continue
        dy, dx = _point((h, w), info["corner"])
        oy, ox = info["point"][0]-dy, info["point"][1]-dx
        if oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        owner = np.full((h, w), -1, dtype=np.int8)
        placed = [None] * len(pieces)

        def fits(i, a, r, c):
            ph, pw = a.shape
            if r+ph > h or c+pw > w or r < 0 or c < 0:
                return False
            if np.any(owner[r:r+ph, c:c+pw] >= 0):
                return False
            # The outside of the finished mosaic is an uninterrupted quiet wall.
            if ((r == 0 and np.any(a[0] != wall))
                    or (c == 0 and np.any(a[:, 0] != wall))
                    or (r+ph == h and np.any(a[-1] != wall))
                    or (c+pw == w and np.any(a[:, -1] != wall))):
                return False
            edges = []
            if r:
                edges.append((a[0], board[r-1, c:c+pw], owner[r-1, c:c+pw]))
            if r+ph < h:
                edges.append((a[-1], board[r+ph, c:c+pw], owner[r+ph, c:c+pw]))
            if c:
                edges.append((a[:, 0], board[r:r+ph, c-1], owner[r:r+ph, c-1]))
            if c+pw < w:
                edges.append((a[:, -1], board[r:r+ph, c+pw], owner[r:r+ph, c+pw]))
            for new, old, owners in edges:
                start = 0
                while start < len(owners):
                    j = int(owners[start])
                    end = start+1
                    while end < len(owners) and owners[end] == j:
                        end += 1
                    if j >= 0 and not _join_ok(new[start:end], old[start:end],
                                              pieces[i]["identity"], pieces[j]["identity"],
                                              info, use_registration, use_compass):
                        return False
                    start = end
            return True

        a = pieces[anchor]["array"]
        ar, ac = pieces[anchor]["old"][0]-oy, pieces[anchor]["old"][1]-ox
        if not fits(anchor, a, ar, ac):
            continue
        ph, pw = a.shape
        board[ar:ar+ph, ac:ac+pw] = a
        owner[ar:ar+ph, ac:ac+pw] = anchor
        placed[anchor] = (ar, ac, a)

        def visit(remaining):
            nonlocal nodes
            nodes += 1
            if not remaining:
                out = np.full(canvas, info["background"], dtype=np.int8)
                out[oy:oy+h, ox:ox+w] = board
                if identity_only:
                    # An ablation probe can ignore wall-label differences and
                    # ask whether complete identity-colored cores move too.
                    cores = np.full(canvas, -1, dtype=np.int8)
                    for i, (r, c, a) in enumerate(placed):
                        ph, pw = a.shape
                        cores[oy+r+1:oy+r+ph-1, ox+c+1:ox+c+pw-1] = pieces[i]["identity"]
                    key = cores.tobytes()
                else:
                    key = out.tobytes()
                if key not in solutions:
                    positions = [(r+oy, c+ox, m.copy()) for r, c, m in placed]
                    solutions[key] = {"output": out, "positions": positions,
                                      "origin": (oy, ox), "shape": (h, w)}
                return len(solutions) >= limit
            r, c = map(int, np.argwhere(owner < 0)[0])
            for i in remaining:
                for a in variants[i]:
                    if not fits(i, a, r, c):
                        continue
                    ph, pw = a.shape
                    board[r:r+ph, c:c+pw] = a
                    owner[r:r+ph, c:c+pw] = i
                    placed[i] = (r, c, a)
                    stop = visit(tuple(j for j in remaining if j != i))
                    board[r:r+ph, c:c+pw] = -1
                    owner[r:r+ph, c:c+pw] = -1
                    placed[i] = None
                    if stop:
                        return True
            return False

        if visit(tuple(i for i in range(len(pieces)) if i != anchor)):
            break
    return list(solutions.values()), nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique reciprocal-compass mosaic derived only from input_grid."""
    info = _parse(input_grid)
    answers, _ = _placements(info)
    if len(answers) != 1:
        raise ValueError("input has zero or multiple distinct mosaic outputs")
    return answers[0]["output"].copy()


def _rectangles(labels):
    rects = []
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i)
        r, c = map(int, q.min(0))
        h, w = map(int, q.max(0)-q.min(0)+1)
        if len(q) != h*w:
            raise ValueError("nonrectangular partition")
        rects.append((r, c, h, w))
    return rects


def _seams(rects):
    result = []
    for i, (r, c, h, w) in enumerate(rects):
        for j in range(i+1, len(rects)):
            y, x, ph, pw = rects[j]
            if c+w == x or x+pw == c:
                lo, hi = max(r, y), min(r+h, y+ph)
                if lo < hi:
                    left, right = (i, j) if c+w == x else (j, i)
                    lr, lc, lh, lw = rects[left]
                    rr, rc, rh, rw = rects[right]
                    pairs = [((k, lc+lw-1), (k, rc)) for k in range(lo, hi)]
                    usable = [k for k, (a, b) in enumerate(pairs)
                              if lr < a[0] < lr+lh-1 and rr < b[0] < rr+rh-1]
                    result.append((left, right, pairs, usable))
            if r+h == y or y+ph == r:
                lo, hi = max(c, x), min(c+w, x+pw)
                if lo < hi:
                    top, bottom = (i, j) if r+h == y else (j, i)
                    tr, tc, th, tw = rects[top]
                    br, bc, bh, bw = rects[bottom]
                    pairs = [((tr+th-1, k), (br, k)) for k in range(lo, hi)]
                    usable = [k for k, (a, b) in enumerate(pairs)
                              if tc < a[1] < tc+tw-1 and bc < b[1] < bc+bw-1]
                    result.append((top, bottom, pairs, usable))
    return result


def _layout(rng, count, kind, witness):
    if kind == "branch":
        heights = [int(rng.integers(4, 7)) for _ in range(count-1)]
        if witness:
            heights = [heights[0]] * (count-1)
        wa, wb = int(rng.integers(4, 7)), int(rng.integers(4, 8))
        labels = np.zeros((sum(heights), wa+wb), dtype=np.int8)
        r = 0
        for i, ph in enumerate(heights, 1):
            labels[r:r+ph, wa:] = i
            r += ph
    elif kind == "grid":
        columns = count//2
        widths = [int(rng.integers(4, 8)) for _ in range(columns)]
        heights = [int(rng.integers(4, 8)) for _ in range(2)]
        if witness == "compass":
            side = int(rng.integers(4, 7))
            widths, heights = [side]*columns, [side]*2
        labels = np.zeros((sum(heights), sum(widths)), dtype=np.int8)
        r = 0
        for i, ph in enumerate(heights):
            c = 0
            for j, pw in enumerate(widths):
                labels[r:r+ph, c:c+pw] = i*columns+j
                c += pw
            r += ph
    else:
        h, w = int(rng.integers(10, 18)), int(rng.integers(10, 18))
        rects = [(0, 0, h, w)]
        while len(rects) < count:
            choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, length in enumerate((ph, pw)) if length >= 8]
            if not choices:
                return None
            i, axis = choices[int(rng.integers(len(choices)))]
            r, c, ph, pw = rects.pop(i)
            cut = int(rng.integers(4, (ph if axis == 0 else pw)-3))
            if axis == 0:
                rects.extend(((r, c, cut, pw), (r+cut, c, ph-cut, pw)))
            else:
                rects.extend(((r, c, ph, cut), (r, c+cut, ph, pw-cut)))
        labels = np.zeros((h, w), dtype=np.int8)
        for i, (r, c, ph, pw) in enumerate(rects):
            labels[r:r+ph, c:c+pw] = i
    if any(len(s[3]) < 2 for s in _seams(_rectangles(labels))):
        return None
    return labels


def _draw(rng, labels, palette, witness):
    bg, marker, wall, reg = palette[:4]
    count = int(labels.max())+1
    if witness == "registration":
        ids = [palette[4]] + [palette[5]]*(count-1)
    elif witness == "compass":
        ids = palette[4:4+count]
    else:
        number = int(rng.integers(2, count+1))
        ids = [palette[4+i % number] for i in range(count)]
        rng.shuffle(ids)
    pic = np.full(labels.shape, wall, dtype=np.int8)
    rects = _rectangles(labels)
    for identity, (r, c, h, w) in zip(ids, rects):
        pic[r+1:r+h-1, c+1:c+w-1] = identity
    for i, j, pairs, usable in _seams(rects):
        if witness:
            # Equal words make the compass witness admit a tile swap without
            # identity checks. All words are chiral on their decorated side.
            reg_positions, port_positions = [usable[0]], [usable[-1]]
            if witness == "registration" and rng.integers(2):
                reg_positions, port_positions = port_positions, reg_positions
        else:
            order = [int(k) for k in rng.permutation(usable)]
            npatch = int(rng.integers(1, min(2, len(order)-1)+1))
            port_positions = order[:npatch]
            nreg = int(rng.integers(1, min(2, len(order)-npatch)+1))
            reg_positions = order[npatch:npatch+nreg]
        for k in reg_positions:
            a, b = pairs[k]
            pic[a] = pic[b] = reg
        for k in port_positions:
            a, b = pairs[k]
            pic[a], pic[b] = ids[j], ids[i]
    return pic


def _halo(points):
    return {(r+dr, c+dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _scatter(rng, arrays, anchor, final, marks, canvas):
    for _ in range(12):
        old = [None]*len(arrays)
        old[anchor] = final[anchor]
        r, c = final[anchor]
        h, w = arrays[anchor].shape
        blocked = _halo({(y, x) for y in range(r, r+h) for x in range(c, c+w)} | set(marks))
        order = sorted((i for i in range(len(arrays)) if i != anchor),
                       key=lambda i: -arrays[i].size)
        failed = False
        for i in order:
            h, w = arrays[i].shape
            candidates = [(r, c) for r in range(canvas[0]-h+1)
                          for c in range(canvas[1]-w+1)]
            rng.shuffle(candidates)
            for r, c in candidates:
                cells = {(y, x) for y in range(r, r+h) for x in range(c, c+w)}
                if not cells & blocked:
                    old[i] = (r, c)
                    blocked.update(_halo(cells))
                    break
            else:
                failed = True
                break
        if not failed:
            return old
    return None


def _make_pair(rng, count, kind, corner, witness=None, motion=None):
    for _ in range(100):
        palette = [int(v) for v in rng.permutation(10)]
        labels = _layout(rng, count, kind, witness)
        if labels is None:
            continue
        pic = _draw(rng, labels, palette, witness)
        eligible = list(range(8))
        rng.shuffle(eligible)
        if witness:
            eligible = [t for t in eligible if _pose(labels, t)[_point(_pose(labels, t).shape, corner)] == 0]
        t = eligible[0]
        labels, pic = _pose(labels, t), _pose(pic, t)
        rects = _rectangles(labels)
        anchor = int(labels[_point(labels.shape, corner)])
        finals = [pic[r:r+h, c:c+w].copy() for r, c, h, w in rects]
        sources = []
        for i, a in enumerate(finals):
            pose = 0 if i == anchor else int(rng.integers(8))
            if i != anchor and motion:
                pose = int(rng.integers(1, 4) if motion == "rotation" else rng.integers(4, 8))
            sources.append(_pose(a, pose))
        area = pic.size
        base = int(math.ceil(math.sqrt(area*2.7)))
        min_h = max(15, pic.shape[0]+2, max(a.shape[0] for a in sources)+2, base)
        min_w = max(15, pic.shape[1]+2, max(a.shape[1] for a in sources)+2, base)
        if max(min_h, min_w) > 30:
            continue
        canvas = (int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))
        low_r, low_c = int(corner[0] == "T"), int(corner[1] == "L")
        high_r = canvas[0]-pic.shape[0]-int(corner[0] == "B")
        high_c = canvas[1]-pic.shape[1]-int(corner[1] == "R")
        origin = (int(rng.integers(low_r, high_r+1)), int(rng.integers(low_c, high_c+1)))
        dy, dx = _point(pic.shape, corner)
        point = origin[0]+dy, origin[1]+dx
        marks = _marker(point, corner)
        final_origins = [(origin[0]+r, origin[1]+c) for r, c, _, _ in rects]
        old = _scatter(rng, sources, anchor, final_origins, marks, canvas)
        if old is None:
            continue
        inp = np.full(canvas, palette[0], dtype=np.int8)
        for a, (r, c) in zip(sources, old):
            inp[r:r+a.shape[0], c:c+a.shape[1]] = a
        for p in marks:
            inp[p] = palette[1]
        out = np.full(canvas, palette[0], dtype=np.int8)
        out[origin[0]:origin[0]+pic.shape[0], origin[1]:origin[1]+pic.shape[1]] = pic
        try:
            info = _parse(inp)
            answers, _ = _placements(info)
            if len(answers) != 1 or not np.array_equal(answers[0]["output"], out):
                continue
            if witness == "registration" and len(_placements(info, use_registration=False)[0]) < 2:
                continue
            if witness == "compass" and len(_placements(info, use_compass=False, identity_only=True)[0]) < 2:
                continue
            if motion and motion not in _moves(info, answers[0]):
                continue
        except ValueError:
            continue
        return {"input": inp, "output": out}
    # No narrower fallback silently replaces the requested teaching/shift case.
    raise RuntimeError("bounded mosaic construction could not certify a pair")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("count", "corner", "topology")))
    train_corners = tuple(str(c) for c in rng.choice(_CORNERS, 2, replace=False))
    if regime == "corner":
        test_corners = tuple(c for c in _CORNERS if c not in train_corners)
    else:
        test_corners = _CORNERS
    plans = [(3, "branch", train_corners[0], "registration", "rotation"),
             (4, "grid", train_corners[1], "compass", "reflection")]
    for _ in range(nt-2):
        n = int(rng.integers(3, 5))
        kind = ("grid" if n == 4 else "branch") if regime == "topology" else str(rng.choice(("branch", "recursive", "grid") if n == 4 else ("branch", "recursive")))
        plans.append((n, kind, str(rng.choice(train_corners)), None, None))
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            if split == "train":
                plan = plans[i]
            else:
                n = 6 if regime == "topology" else int(rng.choice((5, 6))) if regime == "count" else int(rng.choice((4, 6)))
                kind = "grid" if regime == "topology" else str(rng.choice(("branch", "recursive", "grid") if n == 6 or n == 4 else ("branch", "recursive")))
                plan = (n, kind, str(rng.choice(test_corners)), None, None)
            for _ in range(8):
                pair = _make_pair(rng, *plan)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("unable to generate distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _moves(info, answer):
    moves = []
    for i, (r, c, a) in enumerate(answer["positions"]):
        if i == info["anchor"]:
            continue
        source = info["pieces"][i]["array"]
        transforms = [t for t in range(8) if np.array_equal(_pose(source, t), a)]
        moves.append("translation" if 0 in transforms else "rotation" if any(t < 4 for t in transforms) else "reflection")
    return moves


def _inspect(inp, out, ablate=False):
    info = _parse(inp)
    answers, nodes = _placements(info)
    assert len(answers) == 1, "input does not determine one answer"
    answer = answers[0]
    assert np.array_equal(answer["output"], out), "incorrect solved mosaic"
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert not np.any(out == info["marker"])
    for p in _marker(info["point"], info["corner"]):
        assert out[p] == info["background"]
    a = info["anchor"]
    r, c, tile = answer["positions"][a]
    assert (r, c) == info["pieces"][a]["old"]
    assert np.array_equal(tile, info["pieces"][a]["array"])
    cin, cout = (np.bincount(g.ravel(), minlength=10) for g in (inp, out))
    assert cin[info["marker"]] == 3 and cout[info["marker"]] == 0
    assert cout[info["background"]] == cin[info["background"]]+3
    assert all(cin[k] == cout[k] for k in range(10) if k not in (info["background"], info["marker"]))
    occupied = np.zeros(inp.shape, dtype=bool)
    rects = []
    old_cells, new_cells = set(), set()
    for p, (r, c, tile) in zip(info["pieces"], answer["positions"]):
        assert any(np.array_equal(_pose(p["array"], t), tile) for t in range(8))
        h, w = tile.shape
        assert not occupied[r:r+h, c:c+w].any()
        occupied[r:r+h, c:c+w] = True
        rects.append((r, c, h, w))
        new_cells.update((y, x) for y in range(r, r+h) for x in range(c, c+w))
        sr, sc = p["old"]
        sh, sw = p["array"].shape
        old_cells.update((y, x) for y in range(sr, sr+sh) for x in range(sc, sc+sw))
    oy, ox = answer["origin"]
    h, w = answer["shape"]
    assert occupied[oy:oy+h, ox:ox+w].all() and occupied.sum() == h*w
    assert np.array_equal(occupied, out != info["background"])
    seams = _seams(rects)
    # Measure contacts from the actual solved rectangles, not requested modes.
    subedges = 0
    for i, j, pairs, _ in seams:
        vertical = pairs[0][0][0] != pairs[-1][0][0]
        lengths = (rects[i][2], rects[j][2]) if vertical else (rects[i][3], rects[j][3])
        subedges += len(pairs) < max(lengths)
        first = np.array([out[p] for p, _ in pairs])
        second = np.array([out[p] for _, p in pairs])
        assert _join_ok(first, second, info["pieces"][i]["identity"],
                        info["pieces"][j]["identity"], info)
    ids = [p["identity"] for p in info["pieces"]]
    record = {"canvas": inp.shape, "target": answer["shape"], "count": len(rects),
              "corner": info["corner"], "background": info["background"],
              "identity_count": len(set(ids)), "repeated_ids": len(set(ids)) < len(ids),
              "subedges": int(subedges), "cycles": len(seams)-len(rects)+1,
              "spanning_tiles": sum(ph == h or pw == w for _, _, ph, pw in rects),
              "moves": _moves(info, answer), "overlap": len(old_cells & new_cells),
              "nodes": nodes, "dimensions": tuple((h, w) for _, _, h, w in rects)}
    if ablate:
        record["registration_needed"] = len(_placements(info, use_registration=False)[0]) > 1
        record["compass_needed"] = len(_placements(info, use_compass=False)[0]) > 1
        record["compass_moves_cores"] = len(_placements(info, use_compass=False, identity_only=True)[0]) > 1
    return record


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            records[split].append(_inspect(inp, out, ablate=split == "train"))
    train, test = records["train"], records["test"]
    assert any(p["registration_needed"] for p in train), "missing A witness"
    assert any(p["compass_needed"] for p in train), "missing B witness"
    assert any(p["compass_moves_cores"] for p in train), "compass must determine core placement too"
    assert any("rotation" in p["moves"] for p in train)
    assert any("reflection" in p["moves"] for p in train)
    corners = {p["corner"] for p in train}
    assert all(p["count"] > max(t["count"] for t in train)
               or p["corner"] not in corners
               or p["cycles"] > max(t["cycles"] for t in train)
               for p in test), "missing train/test shift"


def _key(puzzle):
    return tuple((p[k].shape, p[k].tobytes()) for s in ("train", "test")
                 for p in puzzle[s] for k in ("input", "output"))


def self_test():
    """Bounded semantic and diversity checks over exactly 42 default seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_key(p) for p in puzzles}) == 42
    assert _key(generate_puzzle(17)) == _key(puzzles[17])
    records = [_inspect(p["input"], p["output"]) for puzzle in puzzles
               for split in ("train", "test") for p in puzzle[split]]
    assert {p["count"] for p in records} == {3, 4, 5, 6}
    assert {p["corner"] for p in records} == set(_CORNERS)
    assert len({p["canvas"] for p in records}) >= 25
    assert len({p["target"] for p in records}) >= 25
    assert len({p["dimensions"] for p in records}) >= 60
    assert any(p["cycles"] >= 2 for p in records)
    assert any(p["subedges"] >= 2 for p in records)
    assert any(p["repeated_ids"] for p in records)
    assert any(not p["repeated_ids"] for p in records)
    assert set(m for p in records for m in p["moves"]) == {"translation", "rotation", "reflection"}
    # The same input rule is equivariant to a whole-scene D4 transform and to
    # any bijective palette permutation. Counts and appearance are not clues.
    pair = puzzles[0]["test"][0]
    for t in range(8):
        assert np.array_equal(solve(_pose(pair["input"], t)), _pose(pair["output"], t))
    palette = np.random.default_rng(311).permutation(10)
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    for a, b in ((2, 1), (10, 4)):
        p = generate_puzzle(51+a, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(7, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for corruption in ("marker", "color", "anchor", "duplicate"):
        damaged = copy.deepcopy(puzzles[0])
        pair = damaged["test"][0]
        info = _parse(pair["input"])
        if corruption == "marker":
            pair["output"][_marker(info["point"], info["corner"])[0]] = info["marker"]
        elif corruption == "color":
            r, c = np.argwhere(pair["output"] != info["background"])[0]
            pair["output"][r, c] = (int(pair["output"][r, c])+1) % 10
        elif corruption == "anchor":
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        else:
            damaged["train"][1] = copy.deepcopy(damaged["train"][0])
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corruption accepted: "+corruption)
    return {"seeds": 42, "pairs": len(records),
            "canvas_shapes": len({p["canvas"] for p in records}),
            "target_shapes": len({p["target"] for p in records}),
            "tile_dimension_sets": len({p["dimensions"] for p in records}),
            "piece_counts": sorted({p["count"] for p in records}),
            "cycle_ranks": sorted({p["cycles"] for p in records}),
            "pairs_with_subedges": sum(bool(p["subedges"]) for p in records),
            "pairs_with_repeated_identities": sum(p["repeated_ids"] for p in records),
            "moves": {m: sum(p["moves"].count(m) for p in records)
                      for m in ("translation", "rotation", "reflection")},
            "max_search_nodes": max(p["nodes"] for p in records)}
