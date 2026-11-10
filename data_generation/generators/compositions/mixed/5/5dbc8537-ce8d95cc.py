"""Complete a rectangular-tile stencil, then contract equal consecutive vectors.

Only NumPy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test. The solver knows no sampling plan.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _contract(grid):
    """Both equality tests refer to full vectors of the completed stencil."""
    rows = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    cols = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[np.ix_(rows, cols)].copy()


def _runs(grid, axis):
    vectors = grid if axis == 0 else grid.T
    starts = np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]
    return tuple(int(n) for n in np.diff(np.r_[starts, len(vectors)]))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for sy, sx in np.argwhere(mask):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        stack, cells = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _tiles(tray, vacancy):
    tiles = []
    for cells in _components(tray != vacancy):
        ys, xs = zip(*cells)
        tile = tray[min(ys):max(ys) + 1, min(xs):max(xs) + 1]
        if len(cells) != tile.size or np.any(tile == vacancy):
            raise ValueError("Source objects must be filled rectangles")
        tiles.append(tile.copy())
    if not tiles:
        raise ValueError("Empty source tray")
    return tiles


def _window_sums(mask, h, w):
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


class _SearchLimit(ValueError):
    pass


class _AmbiguousCover(ValueError):
    pass


def _cover(mask, tiles, max_nodes=16000, reject_ambiguity=False):
    """Exhaustively establish one rendered completion with finite tile inventory.

    Return None for no cover or multiple rendered completions. Different tilings
    that paint identical pixels are allowed. Budget exhaustion is never treated
    as evidence of uniqueness. Dead states store only genuinely impossible states.
    """
    groups, lookup = [], {}
    for tile in tiles:
        k = _key(tile)
        if k in lookup:
            groups[lookup[k]][1] += 1
        else:
            lookup[k] = len(groups)
            groups.append([tile, 1])
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return None
    indices = np.full(mask.shape, -1, dtype=int)
    for i, p in enumerate(coords):
        indices[p] = i
    by_cell = [[] for _ in coords]
    placements = []
    for g, (tile, _) in enumerate(groups):
        h, w = tile.shape
        if h > mask.shape[0] or w > mask.shape[1]:
            continue
        for y, x in np.argwhere(_window_sums(mask, h, w) == h * w):
            y, x = int(y), int(x)
            cells = tuple(int(i) for i in indices[y:y + h, x:x + w].flat)
            bits = sum(1 << i for i in cells)
            p = len(placements)
            placements.append((g, bits, y, x))
            for i in cells:
                by_cell[i].append(p)
    if any(not options for options in by_cell):
        return None
    full = (1 << len(coords)) - 1
    remaining = [count for _, count in groups]
    path, dead = [], set()
    first, first_path = None, None
    ambiguous = False
    nodes = 0

    def visit(covered):
        nonlocal first, first_path, ambiguous, nodes
        nodes += 1
        if nodes > max_nodes:
            raise _SearchLimit("Exact-cover uniqueness search exceeded its node bound")
        if covered == full:
            painted = np.full(mask.shape, -1, dtype=np.int16)
            for pi in path:
                g, _, y, x = placements[pi]
                tile = groups[g][0]
                h, w = tile.shape
                painted[y:y + h, x:x + w] = tile
            if first is None:
                first, first_path = painted, tuple(path)
            elif not np.array_equal(first, painted):
                ambiguous = True
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i, options in enumerate(by_cell):
            if covered & (1 << i):
                continue
            legal = [p for p in options if remaining[placements[p][0]]
                     and not (covered & placements[p][1])]
            if not legal:
                dead.add(state)
                return False
            if best is None or len(legal) < len(best):
                best = legal
                if len(best) == 1:
                    break
        found = False
        for p in best:
            g, bits, _, _ = placements[p]
            remaining[g] -= 1
            path.append(p)
            found |= visit(covered | bits)
            path.pop()
            remaining[g] += 1
            if ambiguous:
                return True
        if not found:
            dead.add(state)
        return found

    visit(0)
    if ambiguous and reject_ambiguity:
        raise _AmbiguousCover("A semantic panel interpretation has multiple rendered completions")
    if first is None or ambiguous:
        return None
    used = Counter(placements[p][0] for p in first_path)
    return {"painted": first, "placements": tuple(placements[p] for p in first_path),
            "groups": groups, "used": used, "nodes": nodes, "dead": len(dead)}


def _panel_candidates(grid):
    """All semantic interpretations, with no expected output dimensions."""
    for axis in (0, 1):
        for cut in range(1, grid.shape[axis]):
            a, b = ((grid[:cut], grid[cut:]) if axis == 0 else
                    (grid[:, :cut], grid[:, cut:]))
            for stencil, tray, side in (
                    (a, b, "top" if axis == 0 else "left"),
                    (b, a, "bottom" if axis == 0 else "right")):
                colors = np.unique(stencil)
                if len(colors) != 2:
                    continue
                for board, vacancy in (colors, colors[::-1]):
                    if np.any(tray == board) or not np.any(tray == vacancy):
                        continue
                    try:
                        tiles = _tiles(tray, vacancy)
                    except ValueError:
                        continue
                    mask = stencil == vacancy
                    if sum(t.size for t in tiles) >= int(mask.sum()):
                        yield stencil, tray, int(board), int(vacancy), tiles, side


def _analyze(input_grid):
    grid = _grid(input_grid)
    answers = []
    for stencil, tray, board, vacancy, tiles, side in _panel_candidates(grid):
        mask = stencil == vacancy
        cover = _cover(mask, tiles, reject_ambiguity=True)
        if cover is None:
            continue
        completed = stencil.copy()
        completed[mask] = cover["painted"][mask]
        answers.append({"stencil": stencil, "tray": tray, "board": board,
                        "vacancy": vacancy, "mask": mask, "tiles": tiles,
                        "side": side, "cover": cover, "completed": completed,
                        "output": _contract(completed)})
    if len(answers) != 1:
        raise ValueError("Input must have one semantic panel split and unique rendered exact cover")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer both panels, exact-cover vacancies, and contract completed vectors."""
    return _analyze(input_grid)["output"]


def _grow(rng, h, w, count):
    """Sample rectangles with arbitrary partial edge joins and optional islands."""
    mask = np.zeros((h, w), dtype=bool)
    rects = []
    max_h, max_w = min(5, h - 1), min(4, w - 1)
    for i in range(count):
        if i == 0:
            th, tw = int(rng.integers(3, max_h + 1)), int(rng.integers(2, max_w + 1))
        elif rects and rng.random() < .17:
            th, tw = rects[int(rng.integers(len(rects)))][2:]
        else:
            th, tw = int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_w + 1))
        candidates = set()
        for _ in range(65):
            if i == 0 or rng.random() < .10:
                y, x = int(rng.integers(h - th + 1)), int(rng.integers(w - tw + 1))
            else:
                yy, xx, hh, ww = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = yy - th if side == 0 else yy + hh
                    x = xx + int(rng.integers(1 - tw, ww))
                else:
                    x = xx - tw if side == 2 else xx + ww
                    y = yy + int(rng.integers(1 - th, hh))
            if 0 <= y <= h - th and 0 <= x <= w - tw and not mask[y:y + th, x:x + tw].any():
                candidates.add((y, x))
        if not candidates:
            return None
        candidates = sorted(candidates)  # No hash-order dependence of seeded draws.
        y, x = candidates[int(rng.integers(len(candidates)))]
        mask[y:y + th, x:x + tw] = True
        rects.append((y, x, th, tw))
    return mask, rects


def _pattern(rng, h, w, palette):
    mode = int(rng.integers(6))
    if mode == 0 or len(palette) == 1:
        return np.full((h, w), int(rng.choice(palette)), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, (h, 1)), w, axis=1).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(palette, (1, w)), h, axis=0).astype(np.int8)
    if mode == 3:
        a, b = rng.choice(palette, 2, replace=False)
        return np.where(np.indices((h, w)).sum(0) % 2, a, b).astype(np.int8)
    if mode == 4:
        a, b = rng.choice(palette, 2, replace=False)
        p = np.full((h, w), a, dtype=np.int8)
        p[int(rng.integers(h)), int(rng.integers(w))] = b
        return p
    return rng.choice(palette, (h, w)).astype(np.int8)


def _separated_payload_repeat(completed, mask, palette, rng):
    """Install A/B/A full vectors on an existing constant-silhouette band."""
    candidates = []
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        for i in range(len(m) - 2):
            if m[i].any() and np.array_equal(m[i], m[i + 1]) and np.array_equal(m[i], m[i + 2]):
                candidates.append((axis, i))
    if not candidates:
        return False
    axis, i = candidates[int(rng.integers(len(candidates)))]
    a = completed if axis == 0 else completed.T
    m = mask if axis == 0 else mask.T
    colors = rng.choice(palette, int(m[i].sum())).astype(np.int8)
    a[i, m[i]] = colors
    a[i + 2, m[i]] = colors
    a[i + 1, m[i]] = colors
    j = int(np.flatnonzero(m[i])[0])
    a[i + 1, j] = int(rng.choice([c for c in palette if c != a[i, j]]))
    return True


def _lengths(rng, mask, axis, cap, no_op=False):
    n = mask.shape[axis]
    lengths = np.ones(n, dtype=int)
    if no_op:
        return lengths
    # Force a payload-bearing vector to repeat; background-only repeats do not
    # suffice as the training evidence for the inherited compression concept.
    occupied = np.flatnonzero(mask.any(axis=1 - axis))
    lengths[int(rng.choice(occupied))] += 1
    extra = int(rng.integers(0, min(cap - n - 1, n) + 1))
    for _ in range(extra):
        lengths[int(rng.integers(n))] += 1
    return lengths


def _scatter(rng, tiles, height, max_width, vacancy):
    """Individually scatter rectangles, with at least one background cell between."""
    minimum = max(t.shape[1] for t in tiles)
    if minimum > max_width or max(t.shape[0] for t in tiles) > height:
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for attempt in range(12):
        low = max(minimum, min(max_width, int(sum(t.size for t in tiles) / height * 1.6) + 3))
        width = max_width if attempt >= 8 else int(rng.integers(low, max_width + 1))
        occupied = np.zeros((height, width), dtype=bool)
        tray = np.full((height, width), vacancy, dtype=np.int8)
        for i in order:
            tile = tiles[i]
            th, tw = tile.shape
            forbidden = occupied.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y0, y1 = max(0, dy), min(height, height + dy)
                    x0, x1 = max(0, dx), min(width, width + dx)
                    forbidden[y0:y1, x0:x1] |= occupied[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            choices = np.argwhere(_window_sums(forbidden, th, tw) == 0)
            if not len(choices):
                break
            y, x = map(int, choices[int(rng.integers(len(choices)))])
            occupied[y:y + th, x:x + tw] = True
            tray[y:y + th, x:x + tw] = tile
        else:
            return tray
    return None


def _orient(stencil, tray, completed, side, rng):
    # An isometry acts on the entire input and completion together; individual
    # tiles are still translated without changing their visible orientation.
    inp = np.concatenate((stencil, tray) if side in ("left", "top") else (tray, stencil), axis=1)
    if side in ("top", "bottom"):
        inp, completed = inp.T, completed.T
    if bool(rng.integers(2)):
        if side in ("left", "right"):
            inp, completed = inp[::-1], completed[::-1]
        else:
            inp, completed = inp[:, ::-1], completed[:, ::-1]
    return inp.copy(), completed.copy()


def _features(a):
    completed, mask = a["completed"], a["mask"]
    unsafe, payload_repeats, separated = [], [], []
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        c = completed if axis == 0 else completed.T
        unsafe.append(any(np.array_equal(m[i], m[i - 1]) and not np.array_equal(c[i], c[i - 1])
                          for i in range(1, len(m))))
        payload_repeats.append(any(m[i].any() and np.array_equal(c[i], c[i - 1])
                                   for i in range(1, len(m))))
        representatives = c[np.r_[True, np.any(c[1:] != c[:-1], axis=1)]]
        seen = set()
        has = False
        for row in representatives:
            if np.any(row != a["board"]):
                k = row.tobytes()
                has |= k in seen
                seen.add(k)
        separated.append(has)
    cover = a["cover"]
    owner = np.full(mask.shape, -1, dtype=int)
    used_tiles = []
    for i, (g, _, y, x) in enumerate(cover["placements"]):
        tile = cover["groups"][g][0]
        h, w = tile.shape
        owner[y:y + h, x:x + w] = i
        used_tiles.append(tile)
    components = _components(mask)
    cluster = max(len({int(owner[y, x]) for y, x in cells}) for cells in components)
    nonrect = any(len(cells) != (max(y for y, _ in cells) - min(y for y, _ in cells) + 1)
                  * (max(x for _, x in cells) - min(x for _, x in cells) + 1) for cells in components)
    return {"side": a["side"], "roles": (a["board"], a["vacancy"]),
            "board_shape": completed.shape, "output_shape": a["output"].shape,
            "row_runs": _runs(completed, 0), "column_runs": _runs(completed, 1),
            "tiles": len(used_tiles), "surplus": len(a["tiles"]) - len(used_tiles),
            "cluster": cluster, "nonrect": nonrect, "components": len(components),
            "unsafe": tuple(unsafe), "payload_repeats": tuple(payload_repeats),
            "separated": tuple(separated), "nodes": cover["nodes"], "dead": cover["dead"],
            "asymmetric": any(not np.array_equal(t, t[::-1, ::-1]) for t in used_tiles),
            "repeated_tiles": any(n > 1 for n in cover["used"].values()),
            "palette": len({int(c) for t in a["tiles"] for c in t.flat}),
            "coverage": float(mask.mean())}


def _make_pair(rng, side, roles, anchor, surplus, test=False, fallback=False):
    board, vacancy = roles
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(2 if anchor else 1, 7))]
    if fallback:
        mask = np.zeros((8, 9), dtype=bool)
        rects = [(1, 1, 3, 4), (4, 1, 2, 2), (4, 3, 1, 3)]
        for y, x, h, w in rects:
            mask[y:y + h, x:x + w] = True
        if len(palette) == 1:
            palette.append(next(c for c in range(10) if c not in roles and c not in palette))
    else:
        h, w = int(rng.integers(7, 13)), int(rng.integers(5, 10))
        count = int(rng.integers(3 if not test else 5, 7 if not test else 10))
        grown = _grow(rng, h, w, count)
        if grown is None:
            return None
        mask, rects = grown
    completed = np.full(mask.shape, board, dtype=np.int8)
    # Equal dimensions start with equal patterns; full uniqueness is checked again
    # after cross-tile vector constraints and all nonuniform repetitions are added.
    patterns = {}
    for y, x, h, w in rects:
        if (h, w) not in patterns:
            patterns[h, w] = _pattern(rng, h, w, palette)
        completed[y:y + h, x:x + w] = patterns[h, w]
    if anchor and not _separated_payload_repeat(completed, mask, palette, rng):
        return None
    no_op = None if anchor else (int(rng.integers(2)) if rng.random() < .16 else None)
    row_lengths = _lengths(rng, mask, 0, 24, no_op == 0)
    col_lengths = _lengths(rng, mask, 1, 13, no_op == 1)
    rows, cols = np.r_[0, np.cumsum(row_lengths)], np.r_[0, np.cumsum(col_lengths)]
    completed = np.repeat(np.repeat(completed, row_lengths, 0), col_lengths, 1)
    mask = np.repeat(np.repeat(mask, row_lengths, 0), col_lengths, 1)
    tiles = [completed[rows[y]:rows[y + h], cols[x]:cols[x + w]].copy() for y, x, h, w in rects]
    try:
        covering = _cover(mask, tiles, max_nodes=3000)
    except _SearchLimit:
        return None
    if covering is None or not np.array_equal(covering["painted"][mask], completed[mask]):
        return None
    # Surplus is optional but visibly demonstrated in training whenever enabled.
    # Prefer locally fitting alternatives and certify that the full packing still
    # forces the same rendering. Never use compression to resolve tile ambiguity.
    if surplus:
        used_dims = {t.shape for t in tiles}
        candidates = [(h, w) for h in range(1, min(7, mask.shape[0]) + 1)
                      for w in range(1, min(7, 30 - mask.shape[1]) + 1) if (h, w) not in used_dims]
        rng.shuffle(candidates)
        candidates.sort(key=lambda d: not (d[0] <= mask.shape[0] and d[1] <= mask.shape[1]
                                           and np.any(_window_sums(mask, *d) == d[0] * d[1])))
        for h, w in candidates[:22]:
            decoy = _pattern(rng, h, w, palette)
            try:
                checked = _cover(mask, tiles + [decoy], max_nodes=3000)
            except _SearchLimit:
                continue
            if checked is not None and np.array_equal(checked["painted"], covering["painted"]):
                tiles.append(decoy)
                break
        else:
            return None
    tray = _scatter(rng, tiles, completed.shape[0], 30 - completed.shape[1], vacancy)
    if tray is None:
        return None
    stencil = np.full(mask.shape, board, dtype=np.int8)
    stencil[mask] = vacancy
    inp, completed = _orient(stencil, tray, completed, side, rng)
    try:
        a = _analyze(inp)
    except ValueError:
        return None
    if not np.array_equal(a["completed"], completed):
        return None
    f = _features(a)
    if max(f["row_runs"] + f["column_runs"]) == 1:
        return None
    if anchor:
        if not (all(f["payload_repeats"]) and any(f["separated"]) and any(f["unsafe"])
                and f["asymmetric"] and f["cluster"] >= 3 and f["nonrect"]):
            return None
    return {"input": inp, "output": _contract(completed)}


def _checked_int(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"Illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_int(seed, "seed", 0)
    num_train = _checked_int(num_train, "num_train", 2, 10)
    num_test = _checked_int(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    sides = [str(rng.choice(["left", "right"])), str(rng.choice(["top", "bottom"]))]
    if bool(rng.integers(2)):
        sides.reverse()
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    roles = [(a, b) for a in range(10) for b in range(10) if a != b]
    roles = [roles[int(i)] for i in rng.permutation(len(roles))[:num_train + num_test]]
    allow_surplus = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(num_train + num_test):
        training = i < num_train
        index = i if training else i - num_train
        side = sides[index % 2] if training else opposite[sides[index % 2]]
        # The first two demonstrations expose the entire causal dependency, even
        # for the minimum legal training count. No convention is hidden in a seed.
        anchor = training and index < 2
        surplus = allow_surplus and (i == 0 or rng.random() < .4)
        pair = None
        for attempt in range(100):
            pair = _make_pair(rng, side, roles[i], anchor, surplus, not training)
            if pair is not None and _key(pair["input"]) not in seen:
                break
        else:
            # Finite fallback uses the same rule, palette freedom, orientation,
            # vector constraints, and full uniqueness checks as ordinary samples.
            for attempt in range(40):
                pair = _make_pair(rng, side, roles[i], True, surplus, fallback=True)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Bounded construction failed")
        seen.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned arrays only, including cover uniqueness and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    features, seen = [], set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        a = _analyze(inp)
        assert np.array_equal(a["output"], out)
        assert not np.any(a["completed"] == a["vacancy"])
        assert np.all(a["completed"][~a["mask"]] == a["board"])
        assert out.shape != a["completed"].shape  # Omitting B changes every answer.
        assert not np.array_equal(_contract(a["stencil"]), out)  # Omitting A.
        # Contraction is lossless together with its maximal run lengths.
        rebuilt = np.repeat(np.repeat(out, _runs(a["completed"], 0), 0),
                            _runs(a["completed"], 1), 1)
        assert np.array_equal(rebuilt, a["completed"])
        assert np.array_equal(_contract(out), out)
        # Both axis orders give exactly the independently defined contraction.
        row_only = a["completed"][np.r_[True, np.any(a["completed"][1:] != a["completed"][:-1], axis=1)]]
        cols = np.r_[True, np.any(row_only[:, 1:] != row_only[:, :-1], axis=0)]
        assert np.array_equal(row_only[:, cols], out)
        features.append(_features(a))
    training, testing = features[:nt], features[nt:]
    for f in training[:2]:
        assert all(f["payload_repeats"]) and any(f["unsafe"]) and any(f["separated"])
        assert f["cluster"] >= 3 and f["nonrect"] and f["asymmetric"]
    assert not any(f["surplus"] for f in testing) or any(f["surplus"] for f in training)
    assert {f["side"] for f in training}.isdisjoint({f["side"] for f in testing})
    assert len({f["roles"] for f in features}) == len(features)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """A bounded 42-seed sweep; retain hashes/features, never stored grid dumps."""
    signatures, in_shapes, out_shapes, layouts, sides = set(), set(), set(), set(), set()
    stats = Counter()
    palettes, used_counts = set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for pair in p["train"] + p["test"]:
            a = _analyze(pair["input"])
            f = _features(a)
            in_shapes.add(pair["input"].shape)
            out_shapes.add(pair["output"].shape)
            sides.add(f["side"])
            palettes.add(f["palette"])
            used_counts.add(f["tiles"])
            layouts.add((f["board_shape"], f["row_runs"], f["column_runs"],
                         _key(a["mask"])))
            stats["pairs"] += 1
            for name in ("surplus", "nonrect", "repeated_tiles", "asymmetric"):
                stats[name] += bool(f[name])
            stats["both_shrink"] += min(max(f["row_runs"]), max(f["column_runs"])) > 1
            stats["separated_payload_repeat"] += any(f["separated"])
            stats["premature_compression_changes_vectors"] += any(f["unsafe"])
            stats["cluster_at_least_5"] += f["cluster"] >= 5
            stats["local_branch_rejection"] += f["dead"] > 0
    assert len(signatures) == 42 and len(in_shapes) >= 30 and len(out_shapes) >= 15
    assert len(layouts) >= 150 and sides == {"left", "right", "top", "bottom"}
    assert len(palettes) >= 4 and len(used_counts) >= 4
    assert stats["both_shrink"] >= 84 and stats["separated_payload_repeat"] >= 84
    assert stats["premature_compression_changes_vectors"] >= 84 and stats["cluster_at_least_5"] >= 10
    assert stats["surplus"] > 10 and stats["local_branch_rejection"] > 0
    assert _signature(generate_puzzle(123, 5, 2)) == _signature(generate_puzzle(123, 5, 2))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(2026 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Small semantic cases exercise finite copies and rendering-level ambiguity.
    mask = np.ones((1, 4), dtype=bool)
    t = np.array([[2, 3]], dtype=np.int8)
    assert _cover(mask, [t]) is None
    assert _cover(mask, [t, t]) is not None
    assert _cover(mask, [t, np.array([[4, 5]])]) is None
    assert _cover(mask, [np.full((1, 4), 2), np.full((1, 2), 2), np.full((1, 2), 2)]) is not None
    c = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3], [1, 1, 2, 1]])
    assert np.array_equal(_contract(c), [[1, 2, 1], [3, 4, 3], [1, 2, 1]])
    # Exercise the bounded fallback, which ordinary seeds need not encounter.
    for side in ("left", "right", "top", "bottom"):
        for surplus in (False, True):
            rng = np.random.default_rng(591)
            for _ in range(40):
                pair = _make_pair(rng, side, (0, 1), True, surplus, fallback=True)
                if pair is not None:
                    a = _analyze(pair["input"])
                    f = _features(a)
                    assert np.array_equal(solve(pair["input"]), pair["output"])
                    assert all(f["payload_repeats"]) and any(f["unsafe"]) and any(f["separated"])
                    assert bool(f["surplus"]) == surplus
                    break
            else:
                raise AssertionError("Fallback failed its bounded exercise")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(in_shapes),
            "output_shapes": len(out_shapes), "layouts": len(layouts),
            "tile_counts": sorted(used_counts), "payload_palette_sizes": sorted(palettes),
            "counts": dict(stats)}
