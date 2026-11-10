"""A folded path probes a periodic wallpaper, then unrolls its observations.

The framed 3x3 binary patch is the wallpaper. At each path cell, relative
displacement from the higher endpoint selects the wallpaper phase. Three
horizontal periods form one output row; only the middle period uses the
current path color. All conventions are fixed and all roles are visible.
"""

from itertools import permutations, product

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_BORDER = np.ones((5, 5), dtype=bool)
_BORDER[1:4, 1:4] = False
_ORDERS = tuple(product((False, True), tuple(permutations(range(3))),
                        tuple(permutations(range(3)))))


def _legal_mask(mask):
    return (mask.shape == (3, 3) and 3 <= int(mask.sum()) <= 8
            and bool(mask.any(axis=1).all())
            and bool(mask[:, 0].any()) and bool(mask[:, 2].any()))


_MASKS = tuple(m for code in range(1, 511)
              if _legal_mask(m := np.array([(code >> k) & 1 for k in range(9)],
                                          dtype=bool).reshape(3, 3)))


def _ordered(mask, order):
    transpose, rows, cols = order
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


_AGREE = np.array([[np.array_equal(m, _ordered(m, o)) for o in _ORDERS]
                   for m in _MASKS], dtype=bool)
_DENSITIES = np.array([int(m.sum()) for m in _MASKS])


def _shift_equal(mask, dr, dc):
    return np.array_equal(mask, mask[np.ix_((np.arange(3) + dr) % 3,
                                            (np.arange(3) + dc) % 3)])


_TRANSLATION_COUNTS = np.array([sum(_shift_equal(m, r, c)
                                    for r in range(3) for c in range(3))
                                for m in _MASKS])


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    """Read roles, the framed motif, and graph order from pixels only."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(counts.argmax())])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("Carrier must be a strict majority")
    frames = []
    for r in range(a.shape[0] - 4):
        for c in range(a.shape[1] - 4):
            block = a[r:r + 5, c:c + 5]
            accent = int(block[0, 0])
            if accent == bg or not np.all(block[_BORDER] == accent):
                continue
            patch = block[1:4, 1:4]
            if np.all((patch == bg) | (patch == accent)):
                frames.append((r, c, accent, patch == accent))
    if len(frames) != 1:
        raise ValueError("Expected exactly one solid 5x5 frame")
    top, left, accent, mask = frames[0]
    if not _legal_mask(mask):
        raise ValueError("Motif must span 3x3, have a gap and occupy each row")
    foreground = a != bg
    foreground = foreground.copy()
    foreground[top:top + 5, left:left + 5] = False
    cells = {tuple(int(x) for x in p) for p in np.argwhere(foreground)}
    if not 2 <= len(cells) <= 30 or any(int(a[p]) == accent for p in cells):
        raise ValueError("Path needs 2..30 cells, distinct from frame color")
    # A one-cell moat separates the frame from every path cell, even diagonally.
    if any(top - 1 <= r <= top + 5 and left - 1 <= c <= left + 5
           for r, c in cells):
        raise ValueError("Path touches the framed patch")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = [p for p in cells if len(neighbors[p]) == 1]
    if (len(ends) != 2 or any(len(v) not in (1, 2) for v in neighbors.values())
            or ends[0][0] == ends[1][0]):
        raise ValueError("Path must be an induced chain with unequal endpoint rows")
    start = min(ends, key=lambda p: p[0])
    path, visited = [], set()
    prev, cur = None, start
    while cur is not None:
        if cur in visited:
            raise ValueError("Cycle in path")
        visited.add(cur)
        path.append(cur)
        following = [p for p in neighbors[cur] if p != prev]
        prev, cur = cur, following[0] if following else None
    if len(path) != len(cells):
        raise ValueError("Disconnected foreground")
    return {"grid": a, "background": bg, "accent": accent, "mask": mask,
            "frame": (top, left), "path": path,
            "colors": [int(a[p]) for p in path]}


def _render(mask, path, colors, bg, accent):
    coords = np.asarray(path, dtype=int)
    delta = coords - coords[0]
    occupancy = mask[delta[:, 0, None] % 3,
                     (delta[:, 1, None] + np.arange(9)[None, :]) % 3]
    paint = np.full((len(path), 9), accent, dtype=np.int8)
    paint[:, 3:6] = np.asarray(colors, dtype=np.int8)[:, None]
    return np.where(occupancy, paint, bg).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _render(info["mask"], info["path"], info["colors"],
                   info["background"], info["accent"])


def _steps(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]


def _scan_orders(path):
    """Concrete alternatives to tracing, used to select teaching witnesses."""
    cols = sorted({c for _, c in path})
    orders = [sorted(path)]
    for columns in (cols, cols[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in columns for p in sorted(
                (p for p in path if p[1] == c),
                reverse=reverse_last and c == columns[-1])])
    return orders


def _fallback_path(length, above):
    # These paths have all four directions and no accidental side contacts.
    segments = ([((-1, 0), 3), ((0, 1), 4), ((1, 0), 6), ((0, -1), 2),
                 ((1, 0), length - 16)] if above else
                [((0, 1), 3), ((1, 0), 4), ((0, -1), 4), ((-1, 0), 2),
                 ((0, -1), 2), ((1, 0), length - 16)])
    path = [(0, 0)]
    for (dr, dc), n in segments:
        for _ in range(n):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    return path


def _make_path(rng, length, style, allow_above, witness=False, need_above=False,
               cover_phases=False):
    persistence = {"gentle": 10.0, "winding": 0.5, "ordinary": 2.0}[style]
    for _ in range(180):
        path, used, prev = [(0, 0)], {(0, 0)}, None
        for _ in range(length - 1):
            candidates, weights = [], []
            for d in _DIRS:
                p = (path[-1][0] + d[0], path[-1][1] + d[1])
                if p in used or sum((p[0] + dr, p[1] + dc) in used
                                   for dr, dc in _DIRS) != 1:
                    continue
                candidates.append((p, d))
                weights.append(persistence if d == prev else 1.0)
            if not candidates:
                break
            w = np.asarray(weights)
            p, prev = candidates[int(rng.choice(len(candidates), p=w / w.sum()))]
            path.append(p)
            used.add(p)
        if len(path) != length:
            continue
        # Geometry is sampled in all orientations; readout still starts higher.
        rotation, reflect = int(rng.integers(4)), int(rng.choice((-1, 1)))
        path = [(r, reflect * c) for r, c in path]
        for _ in range(rotation):
            path = [(c, -r) for r, c in path]
        if path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        above = min(r for r, _ in path) < path[0][0]
        if (above and not allow_above) or (need_above and not above):
            continue
        if max(r for r, _ in path) - min(r for r, _ in path) < 2:
            continue
        if cover_phases and len({((r - path[0][0]) % 3, (c - path[0][1]) % 3)
                                 for r, c in path}) != 9:
            continue
        if witness and (set(_steps(path)) != set(_DIRS)
                        or any(order == path for order in _scan_orders(path))):
            continue
        return path
    if length < 16:
        return [(r, 0) for r in range(length)]
    path = _fallback_path(length, need_above)
    if rng.integers(2):
        path = [(r, -c) for r, c in path]
    return path


def _sample_colors(rng, path, bg, accent, witness):
    available = [c for c in rng.permutation(10) if c not in (bg, accent)]
    ncolors = int(rng.integers(2, min(7, len(available)) + 1))
    palette = [int(c) for c in available[:ncolors]]
    style = str(rng.choice(("fine", "runs", "dominant", "alternate")))
    for _ in range(48):
        seq = []
        for i in range(len(path)):
            if style == "alternate":
                value = palette[i % 2]
            elif seq and rng.random() < {"fine": 0.15, "runs": 0.65,
                                        "dominant": 0.85}[style]:
                value = seq[-1] if style != "dominant" else palette[0]
            else:
                value = int(rng.choice(palette))
            seq.append(value)
        if not witness:
            return seq
        by_cell = dict(zip(path, seq))
        if seq != seq[::-1] and all([by_cell[p] for p in o] != seq
                                    for o in _scan_orders(path)):
            return seq
    # Guaranteed partition: each competing order crosses at least one boundary.
    indices = {p: i for i, p in enumerate(path)}
    cuts = {0, len(path)}
    for order in _scan_orders(path) + [path[::-1]]:
        i = next(i for i, p in enumerate(order) if indices[p] != i)
        cuts.add(min(i, indices[order[i]]) + 1)
    cuts = sorted(cuts)
    return [int(available[i]) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))
            for _ in range(b - a)]


def _embed(rng, path, colors, mask, bg, accent, registrations=None):
    rmin, cmin = min(r for r, _ in path), min(c for _, c in path)
    path = [(r - rmin, c - cmin) for r, c in path]
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    packings = []
    if bw + 6 <= 30:
        packings.append((max(bh, 5), bw + 6))
    if bh + 6 <= 30:
        packings.append((bh + 6, max(bw, 5)))
    h, w = packings[int(rng.integers(len(packings)))]
    h = min(30, h + int(rng.integers(0, 6)))
    w = min(30, w + int(rng.integers(0, 6)))
    while h * w <= 2 * (len(path) + 16 + int(mask.sum())):
        if h <= w and h < 30 or w == 30:
            h += 1
        else:
            w += 1
    placements = [(int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))),
                  (0, 0), (0, w - bw), (h - bh, 0), (h - bh, w - bw)]
    attempts = [(h, w, placements)]
    if registrations is not None:
        # The first two routes have <=21 cells and both spans >=3, so each
        # span is <=19. All these placements fit, with a moat from frames at
        # origins 0..2. The nine phases guarantee a distinguishing placement.
        large_positions = [(r, c) for r in range(9, 12) for c in range(9, 12)]
        rng.shuffle(large_positions)
        attempts.append((30, 30, large_positions))
    for h, w, placements in attempts:
        for top, left in placements:
            translated = [(r + top, c + left) for r, c in path]
            sr, sc = translated[0]
            if registrations is not None:
                absolute, framed = registrations
                if _shift_equal(mask, sr - absolute[0], sc - absolute[1]):
                    continue
            blocked = np.zeros((h, w), dtype=bool)
            for r, c in translated:
                blocked[max(0, r - 1):min(h, r + 2),
                        max(0, c - 1):min(w, c + 2)] = True
            origins = [(r, c) for r in range(h - 4) for c in range(w - 4)
                       if not blocked[r:r + 5, c:c + 5].any()
                       and (registrations is None or not _shift_equal(
                           mask, sr - r - 1 - framed[0], sc - c - 1 - framed[1]))]
            if not origins:
                continue
            fr, fc = origins[int(rng.integers(len(origins)))]
            grid = np.full((h, w), bg, dtype=np.int8)
            for p, color in zip(translated, colors):
                grid[p] = color
            block = grid[fr:fr + 5, fc:fc + 5]
            block[_BORDER] = accent
            block[1:4, 1:4][mask] = accent
            return grid
    raise AssertionError("Constructive packing failed")


def _pick_mask(rng, allowed):
    ids = np.flatnonzero(allowed)
    if not len(ids):
        raise AssertionError("No eligible motif")
    density = int(rng.choice(np.unique(_DENSITIES[ids])))
    return int(rng.choice(ids[_DENSITIES[ids] == density]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not low <= int(n) <= high):
            raise ValueError("Illegal training/test example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("length", "fold", "denser", "sparser")))
    allow_above = bool(rng.integers(2))
    shared_roles = bool(rng.integers(2))
    palette = [int(x) for x in rng.permutation(10)]
    allowed = np.ones(len(_MASKS), dtype=bool)
    if mode == "denser":
        allowed &= _DENSITIES <= 5
    elif mode == "sparser":
        allowed &= _DENSITIES >= 5
    first = _pick_mask(rng, allowed & (_TRANSLATION_COUNTS == 1))
    partners = (allowed & (_DENSITIES != _DENSITIES[first])
                & ((_AGREE & _AGREE[first]).sum(axis=1) == 1))
    second = _pick_mask(rng, partners)
    train_ids = [first, second]
    for _ in range(2, num_train):
        remaining = allowed.copy()
        remaining[train_ids] = False
        train_ids.append(_pick_mask(rng, remaining))
    test_allowed = np.ones(len(_MASKS), dtype=bool)
    if mode == "denser":
        test_allowed &= _DENSITIES >= 6
    elif mode == "sparser":
        test_allowed &= _DENSITIES <= 4
    test_allowed[train_ids] = False
    test_ids = []
    for _ in range(num_test):
        i = _pick_mask(rng, test_allowed)
        test_ids.append(i)
        test_allowed[i] = False
    puzzle, seen = {"train": [], "test": []}, set()
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for index, mid in enumerate(ids):
            witness = index == 0
            if split == "train":
                length = int(rng.integers(18, 22)) if index < 2 else int(rng.integers(7, 22))
                style = "ordinary" if index < 2 else str(rng.choice(("gentle", "ordinary", "winding")))
            else:
                length = int(rng.integers(22, 31)) if mode == "length" else int(rng.integers(16, 27))
                style = "winding" if mode == "fold" else str(rng.choice(("gentle", "ordinary", "winding")))
            above = allow_above and index == 0
            path = _make_path(rng, length, style, allow_above, witness, above,
                              cover_phases=split == "train" and index < 2)
            roles = palette if shared_roles else [int(x) for x in rng.permutation(10)]
            bg, accent = roles[:2]
            colors = _sample_colors(rng, path, bg, accent, witness)
            mask = _MASKS[mid]
            out = _render(mask, path, colors, bg, accent)
            registrations = None
            if split == "train" and index == 1:
                previous = _decode(puzzle["train"][0]["input"])
                sr, sc = previous["path"][0]
                fr, fc = previous["frame"]
                registrations = ((sr, sc), (sr - fr - 1, sc - fc - 1))
            for _ in range(32):
                inp = _embed(rng, path, colors, mask, bg, accent, registrations)
                if _key(inp) not in seen:
                    break
            else:
                # Motifs are distinct within the puzzle, so this is unreachable.
                raise AssertionError("Exhausted distinct placements")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _scalar_expected(info):
    """Cellwise check separate from the vectorized wallpaper renderer."""
    path, mask = info["path"], info["mask"]
    sr, sc = path[0]
    out = np.full((len(path), 9), info["background"], dtype=np.int8)
    for t, (r, c) in enumerate(path):
        for j in range(9):
            if mask[(r - sr) % 3, (c - sc + j) % 3]:
                out[t, j] = info["colors"][t] if 3 <= j < 6 else info["accent"]
    return out


def _trace_witness(info):
    colors, path = info["colors"], info["path"]
    by_cell = dict(zip(path, colors))
    return (colors != colors[::-1]
            and all([by_cell[p] for p in o] != colors for o in _scan_orders(path)))


def _phase_prediction(info, convention):
    """Alternative wallpaper laws for demonstration sufficiency checks."""
    transpose, row_sign, col_sign, scan_sign, ro, co = convention
    mask = info["mask"].T if transpose else info["mask"]
    delta = np.asarray(info["path"]) - np.asarray(info["path"][0])
    occupied = mask[(row_sign * delta[:, 0, None] + ro) % 3,
                    (col_sign * delta[:, 1, None]
                     + scan_sign * np.arange(9)[None, :] + co) % 3]
    paint = np.full(occupied.shape, info["accent"], dtype=np.int8)
    paint[:, 3:6] = np.asarray(info["colors"])[:, None]
    return np.where(occupied, paint, info["background"]).astype(np.int8)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert out.shape == (len(info["path"]), 9)
            assert np.array_equal(out, _scalar_expected(info))
            assert np.array_equal(solve(inp.copy()), out)
            assert not np.array_equal(inp, out)
            # Carrier gaps are identical in all three periods. Outer copies
            # retain motif accent; the center copies every current path color.
            bg, accent = info["background"], info["accent"]
            assert np.array_equal(out[:, :3], out[:, 6:])
            assert np.array_equal(out[:, :3] == bg, out[:, 3:6] == bg)
            assert np.all((out[:, :3] == bg) | (out[:, :3] == accent))
            for t, color in enumerate(info["colors"]):
                assert np.any(out[t, 3:6] == color)
                assert set(int(x) for x in out[t, 3:6]) <= {bg, color}
            # Ablation of B: solid painted periods must change every answer.
            assert np.any(out == bg)
            infos[split].append(info)
    # Teaching checks are separate from input semantics: they are guarantees
    # for generated puzzles, not restrictions on solve's path vocabulary.
    for split in infos:
        assert any(_trace_witness(i) for i in infos[split])
    survivors = set(range(len(_ORDERS)))
    densities = set()
    phase_needed = False
    for info in infos["train"]:
        densities.add(int(info["mask"].sum()))
        actual = _scalar_expected(info)
        survivors &= {k for k, order in enumerate(_ORDERS)
                      if np.array_equal(_render(_ordered(info["mask"], order),
                                                info["path"], info["colors"],
                                                info["background"], info["accent"]), actual)}
        flat = _render(info["mask"], [(0, 0)] * len(info["path"]), info["colors"],
                       info["background"], info["accent"])
        phase_needed |= not np.array_equal(flat, actual)
    assert len(survivors) == 1 and phase_needed
    assert len(densities) >= 2
    for info in infos["train"][:2]:
        sr, sc = info["path"][0]
        assert len({((r - sr) % 3, (c - sc) % 3) for r, c in info["path"]}) == 9
    # Resolve scroll signs, reading direction, transpose, and reset phase from
    # demonstrations. This also rules out two signs that static mask ordering
    # alone does not distinguish (shift the probe, versus mirror the motif).
    phase_laws = list(product((False, True), (1, -1), (1, -1), (1, -1),
                              range(3), range(3)))
    for info in infos["train"]:
        actual = _scalar_expected(info)
        phase_laws = [law for law in phase_laws
                      if np.array_equal(_phase_prediction(info, law), actual)]
    assert phase_laws == [(False, 1, 1, 1, 0, 0)]
    # Even a learned constant offset cannot rescue absolute-canvas or
    # frame-anchored sampling: their registration changes between witnesses.
    absolute, framed = set(range(9)), set(range(9))
    for info in infos["train"]:
        sr, sc = info["path"][0]
        fr, fc = info["frame"]
        for candidates, dr, dc in ((absolute, sr, sc),
                                   (framed, sr - fr - 1, sc - fc - 1)):
            candidates.intersection_update(k for k in range(9)
                                           if _shift_equal(info["mask"], dr + k // 3, dc + k % 3))
    assert not absolute and not framed
    if any(min(r for r, _ in i["path"]) < i["path"][0][0] for i in infos["test"]):
        assert any(min(r for r, _ in i["path"]) < i["path"][0][0]
                   and _trace_witness(i) for i in infos["train"])


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed coverage, count checks, and rule-specific regressions."""
    puzzles, shapes, layouts, motifs, lengths, turns, palettes = set(), set(), set(), set(), set(), set(), set()
    densities, centers, phase_pairs = set(), set(), set()
    above = border_contact = recurrent = horizontal_reversal = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_fingerprint(p))
        if seed in (0, 17, 41):
            assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                i = _decode(pair["input"])
                path = i["path"]
                origin = path[0]
                normalized = tuple((r - origin[0], c - origin[1]) for r, c in path)
                shapes.add(pair["input"].shape)
                layouts.add(normalized)
                motifs.add(i["mask"].tobytes())
                lengths.add(len(path))
                steps = _steps(path)
                turns.add(sum(a != b for a, b in zip(steps, steps[1:])))
                palettes.add((i["background"], i["accent"]))
                densities.add(int(i["mask"].sum()))
                centers.add(bool(i["mask"][1, 1]))
                phase_pairs.update((dr % 3, dc % 3) for dr, dc in normalized)
                above += min(r for r, _ in path) < origin[0]
                horizontal_reversal += (0, 1) in steps and (0, -1) in steps
                runs = [c for t, c in enumerate(i["colors"]) if t == 0 or c != i["colors"][t - 1]]
                recurrent += len(runs) > len(set(runs))
                h, w = pair["input"].shape
                border_contact += any(r in (0, h - 1) or c in (0, w - 1) for r, c in path)
    assert len(puzzles) == 42
    assert len(shapes) >= 50 and len(layouts) >= 120
    assert len(motifs) >= 75 and len(palettes) >= 25
    assert len(lengths) >= 20 and len(turns) >= 10
    assert densities == set(range(3, 9)) and centers == {False, True}
    assert len(phase_pairs) == 9
    assert min(above, border_contact, recurrent, horizontal_reversal) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4200 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # Both constructive safety nets at every witness length.
    rng = np.random.default_rng(8128)
    mask = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=bool)
    for n in range(16, 31):
        for ab in (False, True):
            path = _fallback_path(n, ab)
            colors = _sample_colors(rng, path, 0, 8, True)
            inp = _embed(rng, path, colors, mask, 0, 8)
            assert np.array_equal(solve(inp), _render(mask, path, colors, 0, 8))
    # Literal regression for the specification, with independent object moves.
    route = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)]
    expected = np.array([[int(c) for c in row] for row in
                         ("800100800", "008002008", "880330880",
                          "808404808", "088055088", "080060080")], dtype=np.int8)

    def fixture(frame=(1, 1), offset=(12, 12), points=route):
        inp = np.zeros((24, 24), dtype=np.int8)
        fr, fc = frame
        block = inp[fr:fr + 5, fc:fc + 5]
        block[_BORDER] = 8
        block[1:4, 1:4][mask] = 8
        for color, (r, c) in enumerate(points, 1):
            inp[r + offset[0], c + offset[1]] = color
        return inp

    for inp in (fixture(), fixture(frame=(17, 2)), fixture(offset=(9, 17))):
        assert np.array_equal(solve(inp), expected)
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[expected])
    # Semantic support outside the generator's demonstration length range.
    for n in (2, 3, 5, 30):
        path = [(r, 0) for r in range(n)]
        colors = [1 + r % 2 for r in range(n)]
        inp = _embed(rng, path, colors, mask, 0, 8)
        assert np.array_equal(solve(inp), _render(mask, path, colors, 0, 8))
    invalid_inputs = [fixture(points=[(0, 0), (0, 1), (0, 2)])]
    disconnected = fixture()
    disconnected[22, 22] = 1
    invalid_inputs.append(disconnected)
    branch = fixture()
    branch[13, 12] = 1
    invalid_inputs.append(branch)
    missing_frame = fixture()
    missing_frame[1, 1] = 0
    invalid_inputs.append(missing_frame)
    empty_row = fixture()
    empty_row[3, 2:5] = 0
    invalid_inputs.append(empty_row)
    for inp in invalid_inputs:
        try:
            solve(inp)
        except ValueError:
            continue
        raise AssertionError("Malformed framed-route input accepted")
    # Negative answers: painting through a motif gap, a wrong scroll phase,
    # and changing a preserved outer accent are all rejected.
    for damage in ("gap", "phase", "outer"):
        p = generate_puzzle(901, 2, 1)
        pair = p["test"][0]
        out = pair["output"]
        i = _decode(pair["input"])
        if damage == "gap":
            point = tuple(np.argwhere(out == i["background"])[0])
            out[point] = i["accent"]
        elif damage == "phase":
            out[:] = np.roll(out, 1, axis=1)
        else:
            point = tuple(np.argwhere(out[:, :3] == i["accent"])[0])
            out[point] = i["background"]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("Damaged wallpaper accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
