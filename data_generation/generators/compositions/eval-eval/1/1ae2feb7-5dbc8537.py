"""Registered pulse mosaics: local run clocks constrained by finite packing.

Only Python and NumPy are needed. No file access, generation metadata, or
training examples are used by solve(). All searches and generation retries
are bounded; self_test streams 42 puzzles without storing grids on disk.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(row):
    """Maximal runs, nearest to the canonical right-hand plate edge first."""
    result = []
    x = len(row) - 1
    while x >= 0:
        color = int(row[x])
        end = x
        while x >= 0 and row[x] == color:
            x -= 1
        result.append((end - x, color))
    return result


def _print_plate(plate, paper, phase=0, priority="near"):
    """Cell-wise arithmetic used by inference, independent of slice painting."""
    out = np.full(plate.shape, paper, dtype=np.int8)
    for y, row in enumerate(plate):
        runs = _runs(row)
        for x in range(plate.shape[1]):
            firing = [(i, n, c) for i, (n, c) in enumerate(runs)
                      if (x + phase) % n == 0]
            if not firing:
                continue
            if priority == "short":
                winner = min(firing, key=lambda v: (v[1], v[0]))
            elif priority == "long":
                winner = min(firing, key=lambda v: (-v[1], v[0]))
            elif priority == "far":
                winner = firing[-1]
            else:
                winner = firing[0]
            out[y, x] = winner[2]
    return out


def _paint_plate(plate, paper):
    """Construction uses overwriting of whole periodic slices."""
    result = np.full(plate.shape, paper, dtype=np.int8)
    for y, row in enumerate(plate):
        for length, color in reversed(_runs(row)):
            result[y, ::length] = color
    return result


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in unseen:
                    unseen.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _border(a):
    c = int(a[0, 0])
    if (np.all(a[0] == c) and np.all(a[-1] == c)
            and np.all(a[:, 0] == c) and np.all(a[:, -1] == c)):
        return c
    return None


def _parse(value):
    a = _grid(value)
    candidates = []
    counts = np.bincount(a.ravel(), minlength=10)
    for turn in range(4):
        g = np.rot90(a, turn)
        h, w = g.shape
        for cut in range(3, w - 3):
            marker = int(g[0, cut])
            if counts[marker] != h or not np.all(g[:, cut] == marker):
                continue
            tray, board = g[:, :cut], g[:, cut + 1:]
            paper, wall = _border(tray), _border(board)
            if paper is None or wall is None or len({paper, wall, marker}) != 3:
                continue
            if np.any(tray == wall) or not np.any(board == paper):
                continue
            plates = []
            valid = True
            for comp in _components(tray != paper):
                yy, xx = zip(*comp)
                r, c, rh, cw = min(yy), min(xx), max(yy) - min(yy) + 1, max(xx) - min(xx) + 1
                p = tray[r:r + rh, c:c + cw]
                if p.size != len(comp) or np.any(p == paper):
                    valid = False
                    break
                plates.append(p.copy())
            if not valid or not plates:
                continue
            payload = set(int(c) for p in plates for c in p.flat)
            if payload & {paper, wall, marker}:
                continue
            if not set(map(int, np.unique(board))) <= payload | {paper, wall}:
                continue
            mask = board != wall
            known = mask & (board != paper)
            if not np.any(known) or sum(p.size for p in plates) < int(mask.sum()):
                continue
            candidates.append(dict(turn=turn, tray=tray.copy(), board=board.copy(),
                                   paper=paper, wall=wall, marker=marker,
                                   plates=plates, mask=mask, known=known))
    if len(candidates) != 1:
        raise ValueError("Input must identify one tray, divider, and bordered stencil")
    return candidates[0]


def _cover(info, clues=True, max_nodes=30000):
    """Finite-inventory exact cover; uniqueness means rendered pixels.

    Each search path is kept only in memory. Identical printed matrices are
    grouped, so copies or distinct recipes with identical prints interchange.
    At most two different answers are retained. Budget exhaustion is an error.
    """
    mask, board, paper = info["mask"], info["board"], info["paper"]
    groups, lookup = [], {}
    for plate in info["plates"]:
        image = _print_plate(plate, paper)
        key = _key(image)
        if key in lookup:
            groups[lookup[key]]["count"] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(image=image, plate=plate, count=1))
    coords = np.argwhere(mask)
    cell_index = np.full(mask.shape, -1, dtype=int)
    cell_index[mask] = np.arange(len(coords))
    options = [[] for _ in coords]
    placements = []
    for gi, group in enumerate(groups):
        image = group["image"]
        h, w = image.shape
        for y in range(mask.shape[0] - h + 1):
            for x in range(mask.shape[1] - w + 1):
                if not np.all(mask[y:y + h, x:x + w]):
                    continue
                visible = info["known"][y:y + h, x:x + w]
                if clues and np.any(image[visible] != board[y:y + h, x:x + w][visible]):
                    continue
                ids = tuple(map(int, cell_index[y:y + h, x:x + w].ravel()))
                bits = sum(1 << i for i in ids)
                pi = len(placements)
                placements.append((gi, y, x, bits, ids))
                for i in ids:
                    options[i].append(pi)
    full = (1 << len(coords)) - 1
    remaining = [g["count"] for g in groups]
    rendered = np.full(len(coords), paper, dtype=np.int8)
    path, answers, dead = [], {}, set()
    nodes = 0

    def visit(covered):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("Exact-cover search budget exceeded")
        if covered == full:
            key = rendered.tobytes()
            if key not in answers:
                out = board.copy()
                out[mask] = rendered
                answers[key] = (out, tuple(path))
            return True
        state = (covered, tuple(remaining))
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            choices = [j for j in options[i]
                       if remaining[placements[j][0]] and not covered & placements[j][3]]
            if not choices:
                dead.add(state)
                return False
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        success = False
        for pi in best:
            gi, y, x, bits, ids = placements[pi]
            remaining[gi] -= 1
            rendered[list(ids)] = groups[gi]["image"].ravel()
            path.append((gi, y, x))
            success = visit(covered | bits) or success
            path.pop()
            remaining[gi] += 1
            rendered[list(ids)] = paper
            if len(answers) >= 2:
                break
        if not success:
            dead.add(state)
        return success

    visit(0)
    return list(answers.values()), groups, nodes


def _infer(value):
    info = _parse(value)
    answers, groups, nodes = _cover(info)
    if len(answers) != 1:
        raise ValueError("Stencil needs exactly one printed completion")
    info.update(output=answers[0][0], path=answers[0][1], groups=groups, nodes=nodes)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _infer(input_grid)
    return np.rot90(info["output"], -info["turn"]).copy()


def _row(lengths, colors):
    return np.array([c for n, c in reversed(list(zip(lengths, colors)))
                     for _ in range(n)], dtype=np.int8)


def _recipe(rng, h, w, palette):
    p = np.empty((h, w), dtype=np.int8)
    for y in range(h):
        count = int(rng.integers(1, min(w, 4) + 1))
        cuts = sorted(map(int, rng.choice(range(1, w), count - 1, replace=False)))
        lengths = np.diff([0] + cuts + [w]).tolist()
        colors = [int(rng.choice(palette))]
        for _ in range(1, count):
            colors.append(int(rng.choice([c for c in palette if c != colors[-1]])))
        p[y] = _row(lengths, colors)
    return p


def _lesson(palette):
    a, b, c = palette[:3]
    return np.array([_row([2, 3, 2], [a, b, c]),
                     _row([3, 2, 2], [b, a, c])], dtype=np.int8)


def _grow(rng, dims, shape):
    """Arbitrary edge contacts and offsets, never a fixed silhouette template."""
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    rects = []
    for th, tw in dims:
        candidates = set()
        for _ in range(90):
            if not rects:
                y, x = int(rng.integers(1, h - th)), int(rng.integers(1, w - tw))
            else:
                ry, rx, rh, rw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = ry - th if side == 0 else ry + rh
                    x = rx + int(rng.integers(1 - tw, rw))
                else:
                    x = rx - tw if side == 2 else rx + rw
                    y = ry + int(rng.integers(1 - th, rh))
            if (y < 1 or x < 1 or y + th >= h or x + tw >= w
                    or np.any(occupied[y:y + th, x:x + tw])):
                continue
            candidates.add((y, x))
        if not candidates:
            return None
        candidates = sorted(candidates)
        y, x = candidates[int(rng.integers(len(candidates)))]
        occupied[y:y + th, x:x + tw] = True
        rects.append((y, x, th, tw))
    return occupied, rects


def _scatter(rng, plates, shape):
    occupied = np.zeros(shape, dtype=bool)
    positions = {}
    for i in sorted(range(len(plates)), key=lambda j: -plates[j].size):
        th, tw = plates[i].shape
        legal = []
        for y in range(1, shape[0] - th):
            for x in range(1, shape[1] - tw):
                if not occupied[y - 1:y + th + 1, x - 1:x + tw + 1].any():
                    legal.append((y, x))
        if not legal:
            return None
        y, x = legal[int(rng.integers(len(legal)))]
        positions[i] = (y, x)
        occupied[y:y + th, x:x + tw] = True
    return positions


def _assemble(tray, board, marker, turn):
    inp = np.concatenate((tray, np.full((tray.shape[0], 1), marker, dtype=np.int8), board), axis=1)
    return np.rot90(inp, turn).copy()


def _global_clock(info):
    """Rival: phase measured from each silhouette component, without tile resets."""
    origins = {}
    for comp in _components(info["mask"]):
        origin = min(x for _, x in comp)
        origins.update({cell: origin for cell in comp})
    out = np.full(info["board"].shape, info["wall"], dtype=np.int8)
    for gi, y, x in info["path"]:
        plate = info["groups"][gi]["plate"]
        image = _print_plate(plate, info["paper"], phase=x - origins[(y, x)])
        h, w = image.shape
        out[y:y + h, x:x + w] = image
    return out


def _register(rng, board, completed, plates, paper, wall):
    """Add sparse positive registration dots until the printed answer is unique."""
    mask = board != wall
    possible = np.argwhere(mask & (completed != paper))
    if len(possible) < 4:
        return None
    cap = min(3 * len(plates) + 5, len(possible) // 2)
    # A few scattered dots; additional dots are selected by actual ambiguity.
    take = min(cap, max(2, len(plates)))
    for i in rng.choice(len(possible), take, replace=False):
        y, x = possible[int(i)]
        board[y, x] = completed[y, x]
    info = dict(board=board, mask=mask, paper=paper, wall=wall, plates=plates,
                known=mask & (board != paper))
    for _ in range(cap + 1):
        info["known"] = mask & (board != paper)
        answers, groups, nodes = _cover(info, max_nodes=7000)
        if not answers:
            return None
        rivals = [a for a, _ in answers if not np.array_equal(a, completed)]
        if len(answers) == 1 and not rivals:
            info.update(output=completed, path=answers[0][1], groups=groups, nodes=nodes)
            if np.array_equal(_global_clock(info), completed):
                return None
            return board
        if int(info["known"].sum()) >= cap:
            return None
        rival = rivals[0]
        distinguish = np.argwhere(mask & (completed != paper) & (rival != completed))
        if not len(distinguish):
            # Positive dots cannot disambiguate a strict superset of the ink.
            return None
        y, x = distinguish[int(rng.integers(len(distinguish)))]
        board[y, x] = completed[y, x]
    return None


def _fallback(rng, count, roles, turn, lesson, repeated, surplus):
    """Finite constructive fallback, with explicit per-plate registration codes."""
    paper, wall, marker = roles
    colors = [c for c in range(10) if c not in roles]
    rng.shuffle(colors)
    count = max(count, 3 if repeated else 2)
    h, bw, tw = max(12, 3 * count + 3), 12, 15
    board = np.full((h, bw), wall, dtype=np.int8)
    completed = board.copy()
    tray = np.full((h, tw), paper, dtype=np.int8)
    plates = []
    for i in range(count):
        a, b, c = colors[i % 7], colors[(i + 1) % 7], colors[(i + 2) % 7]
        p = np.array([_row([2, 3, 2], [a, b, c])] * 2, dtype=np.int8)
        if i == 0 and lesson:
            p = _lesson(colors)
        if repeated and i == 2:
            p = plates[1].copy()
        plates.append(p)
        y, x = 1 + 2 * i, 1 + i % 3
        board[y:y + 2, x:x + 7] = paper
        image = _paint_plate(p, paper)
        completed[y:y + 2, x:x + 7] = image
        board[y, x] = image[0, 0]
        sy, sx = 1 + 3 * i, int(rng.integers(1, tw - 7))
        tray[sy:sy + 2, sx:sx + 7] = p
    if surplus:
        tray[-2, -2] = colors[-1]
    inp = _assemble(tray, board, marker, turn)
    info = _infer(inp)
    assert np.array_equal(info["output"], completed)
    assert not np.array_equal(_global_clock(info), completed)
    return dict(input=inp, output=np.rot90(completed, turn).copy())


def _build_pair(rng, count, turn, lesson=False, repeated=False, surplus=False):
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    paper, wall, marker = roles
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    count = max(count, 3 if repeated else 2)
    for _ in range(70):
        h, bw = int(rng.integers(17, 30)), int(rng.integers(10, 16))
        tw = int(rng.integers(11, 30 - bw))
        max_width = min(9, bw - 2, tw - 2)
        primary = _lesson(palette) if lesson else _recipe(
            rng, int(rng.integers(1, 4)), int(rng.integers(3, max_width + 1)), palette)
        second = _recipe(rng, *primary.shape, palette)
        if np.array_equal(_paint_plate(primary, paper), _paint_plate(second, paper)):
            continue
        plates = [primary, second]
        if repeated:
            plates.append(second.copy())
        while len(plates) < count:
            plates.append(_recipe(rng, int(rng.integers(1, 5)),
                                  int(rng.integers(2, max_width + 1)), palette))
        grown = _grow(rng, [p.shape for p in plates], (h, bw))
        if grown is None:
            continue
        mask, rects = grown
        board = np.full((h, bw), wall, dtype=np.int8)
        board[mask] = paper
        completed = np.full((h, bw), wall, dtype=np.int8)
        for p, (y, x, th, twidth) in zip(plates, rects):
            completed[y:y + th, x:x + twidth] = _paint_plate(p, paper)
        # Optional surplus may fit locally; only the full registered packing
        # decides whether it can be used. It is never a same-size label trick.
        if surplus:
            plates.append(_recipe(rng, int(rng.integers(1, 4)), int(rng.integers(2, 7)), palette))
        positions = _scatter(rng, plates, (h, tw))
        if positions is None:
            continue
        try:
            registered = _register(rng, board, completed, plates, paper, wall)
        except ValueError:
            continue
        if registered is None:
            continue
        tray = np.full((h, tw), paper, dtype=np.int8)
        for i, (y, x) in positions.items():
            th, twidth = plates[i].shape
            tray[y:y + th, x:x + twidth] = plates[i]
        inp = _assemble(tray, registered, marker, turn)
        try:
            info = _infer(inp)
            naked, _, _ = _cover(info, clues=False, max_nodes=7000)
        except ValueError:
            continue
        if len(naked) != 2 or not np.array_equal(info["output"], completed):
            continue
        if surplus and len(info["path"]) >= len(plates):
            continue
        return dict(input=inp, output=np.rot90(completed, turn).copy())
    return _fallback(rng, count, roles, turn, lesson, repeated, surplus)


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    orientation_shift = bool(rng.integers(2))
    allow_surplus = bool(rng.integers(2))
    lesson_slot, repeat_slot = map(int, rng.choice(nt, 2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            training = split == "train"
            if training:
                turn = (base + (2 * int(rng.integers(2)) if orientation_shift else 0)) % 4
                count = int(rng.integers(3, 5))
            elif orientation_shift:
                turn = (base + int(rng.choice([1, 3]))) % 4
                count = int(rng.integers(3, 6))
            else:
                turn, count = base, int(rng.integers(5, 7))
            lesson = training and i == lesson_slot
            repeated = (training and i == repeat_slot) or (not training and rng.random() < .45)
            surplus = allow_surplus and ((training and i == lesson_slot) or rng.random() < .5)
            for _ in range(8):
                pair = _build_pair(rng, count, turn, lesson, repeated, surplus)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    return puzzle


def _features(info):
    raw = np.full(info["board"].shape, info["wall"], dtype=np.int8)
    counts = Counter()
    short, long, edges = False, False, set()
    lesson_rows = []
    for gi, y, x in info["path"]:
        group = info["groups"][gi]
        p = group["plate"]
        h, w = p.shape
        raw[y:y + h, x:x + w] = p
        counts[gi] += 1
        for row in p:
            runs = _runs(row)
            lesson_rows.append(tuple(n for n, _ in runs))
            for i, (n, c) in enumerate(runs):
                for m, d in runs[i + 1:]:
                    if c == d:
                        continue
                    # Count only winners and a surviving farther pulse.
                    winner = [next((j for j, (l, _) in enumerate(runs) if t % l == 0), -1)
                              for t in range(w)]
                    far_survives = any(j > i and runs[j] == (m, d) for j in winner if j >= 0)
                    wins = any(j == i and t % m == 0 for t, j in enumerate(winner))
                    if wins and far_survives:
                        short |= n < m
                        long |= n > m
                        edges.add((c, d))
    return dict(shape=info["board"].shape, turn=info["turn"],
                used=len(info["path"]), supplied=len(info["plates"]),
                clues=int(info["known"].sum()), ink=int(np.sum(info["mask"] & (info["output"] != info["paper"]))),
                area=int(info["mask"].sum()), nodes=info["nodes"],
                repeat=max(counts.values()) > 1,
                raw_diff=not np.array_equal(raw, info["output"]),
                reset_diff=not np.array_equal(_global_clock(info), info["output"]),
                short=short, long=long, cycle=any((b, a) in edges for a, b in edges),
                dimensions=tuple(sorted({p.shape for p in info["plates"]})),
                periods=tuple(sorted({n for p in info["plates"] for r in p for n, _ in _runs(r)})))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        info = _infer(inp)
        target = np.rot90(out, info["turn"])
        assert np.array_equal(target, info["output"]), "Wrong pulse packing or output crop"
        assert np.all(target[~info["mask"]] == info["wall"]), "Board margin changed"
        assert np.array_equal(target[info["known"]], info["board"][info["known"]]), "Registration dot changed"
        assert len(_components(info["mask"])) == 1, "Generated silhouette must be connected"
        cover_count = np.zeros(info["mask"].shape, dtype=int)
        used = Counter()
        for gi, y, x in info["path"]:
            p = info["groups"][gi]["plate"]
            h, w = p.shape
            cover_count[y:y + h, x:x + w] += 1
            used[gi] += 1
            # Verify individual cell arithmetic, including every silent cell.
            for dy, row in enumerate(p):
                runs = _runs(row)
                for dx in range(w):
                    color = next((c for n, c in runs if dx % n == 0), info["paper"])
                    assert target[y + dy, x + dx] == color
        assert np.array_equal(cover_count, info["mask"].astype(int))
        assert all(n <= info["groups"][gi]["count"] for gi, n in used.items())
        naked, _, _ = _cover(info, clues=False)
        assert len(naked) == 2, "Dots must resolve a real rendered packing ambiguity"
        f = _features(info)
        assert f["raw_diff"] and f["reset_diff"], "Either inherited mechanism became inert"
        assert 0 < 2 * f["clues"] <= f["ink"], "At least half the ink must be inferred"
        features.append(f)
    tr, te = features[:nt], features[nt:]
    assert any(f["short"] and f["long"] and f["cycle"] for f in tr), "Priority lesson absent"
    assert any(f["repeat"] for f in tr), "Finite repeated inventory not taught"
    assert not any(f["used"] < f["supplied"] for f in te) or any(f["used"] < f["supplied"] for f in tr)
    return True


def self_test():
    # Hand arithmetic: opposite priority orders with the same colors.
    p = _lesson([1, 2, 3])
    expected = np.array([[1, 0, 1, 2, 1, 0, 1], [2, 0, 1, 2, 1, 0, 2]], dtype=np.int8)
    assert np.array_equal(_print_plate(p, 0), expected)
    assert np.array_equal(_paint_plate(p, 0), expected)
    for rival in ("short", "long", "far"):
        assert not np.array_equal(_print_plate(p, 0, priority=rival), expected)
    # Inventory: a plate cannot be reused; exchanging identical copies is legal.
    plate = np.array([[2, 2, 1, 1]], dtype=np.int8)
    tiny = dict(mask=np.ones((1, 8), dtype=bool), board=np.zeros((1, 8), dtype=np.int8),
                known=np.zeros((1, 8), dtype=bool), paper=0, plates=[plate])
    assert not _cover(tiny)[0]
    tiny["plates"] = [plate, plate.copy()]
    answers, _, _ = _cover(tiny)
    assert len(answers) == 1 and answers[0][0].tolist() == [[1, 0, 1, 0, 1, 0, 1, 0]]
    signatures, shapes, dims, periods, turns = set(), set(), set(), set(), set()
    used_counts, areas, clue_counts, palettes = set(), set(), set(), set()
    max_nodes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for pair in puzzle["train"] + puzzle["test"]:
            info = _infer(pair["input"])
            f = _features(info)
            digest.update(repr(pair["input"].shape).encode())
            digest.update(pair["input"].tobytes())
            shapes.add(pair["input"].shape)
            dims.update(f["dimensions"])
            periods.update(f["periods"])
            turns.add(f["turn"])
            used_counts.add(f["used"])
            areas.add(f["area"])
            clue_counts.add(f["clues"])
            palettes.add((info["paper"], info["wall"], info["marker"]))
            max_nodes = max(max_nodes, f["nodes"])
        signatures.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            for split in puzzle:
                for a, b in zip(puzzle[split], again[split]):
                    assert all(np.array_equal(a[k], b[k]) for k in a)
            # All rotations and a bijective recoloring commute with solving.
            pair = puzzle["train"][0]
            for turn in range(4):
                assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
            cmap = np.array([3, 8, 1, 9, 0, 6, 4, 2, 7, 5], dtype=np.int8)
            assert np.array_equal(solve(cmap[pair["input"]]), cmap[pair["output"]])
            broken = {s: [{k: g.copy() for k, g in q.items()} for q in puzzle[s]] for s in puzzle}
            broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                pass
            else:
                raise AssertionError("Output mutation accepted")
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(dims) >= 18 and len(areas) >= 50 and len(palettes) >= 100
    assert turns == {0, 1, 2, 3} and set(range(1, 9)) <= periods
    assert {3, 4, 5, 6} <= used_counts and len(clue_counts) >= 5
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Exercise the constructive fallback rather than trusting a dormant branch.
    for turn in range(4):
        q = _fallback(np.random.default_rng(100 + turn), 4, (0, 8, 9), turn, True, True, True)
        assert np.array_equal(solve(q["input"]), q["output"])
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                tile_dimensions=len(dims), silhouette_areas=len(areas), role_palettes=len(palettes),
                directions=sorted(turns), periods=sorted(periods), used_counts=sorted(used_counts),
                clue_counts=sorted(clue_counts), max_search_nodes=max_nodes)
