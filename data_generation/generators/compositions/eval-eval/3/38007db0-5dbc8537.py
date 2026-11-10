"""Replay exceptional motif crops into a partially painted rectangle mosaic.

Only NumPy and the standard library are used. No parent code or examples are
imported. Labels are painted from construction placements; solve reads input only.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid, colors 0..9")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _library(lib, frame):
    rs = np.flatnonzero(np.all(lib == frame, axis=1))
    cs = np.flatnonzero(np.all(lib == frame, axis=0))
    if (len(rs) < 3 or len(cs) < 4 or rs[0] != 0 or cs[0] != 0
            or rs[-1] != lib.shape[0] - 1 or cs[-1] != lib.shape[1] - 1
            or not np.all(np.diff(rs) == rs[1])
            or not np.all(np.diff(cs) == cs[1])):
        raise ValueError("Invalid library lattice")
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    bands, copies = len(rs) - 1, len(cs) - 1
    if ph < 1 or pw < 1 or not 2 <= bands <= 5 or not 3 <= copies <= 4:
        raise ValueError("Invalid library dimensions")
    rows = [[lib[r+1:s, c+1:d].copy() for c, d in zip(cs, cs[1:])]
            for r, s in zip(rs, rs[1:])]
    freq = Counter(_key(p) for row in rows for p in row)
    highest = max(freq.values())
    modes = [key for key, count in freq.items() if count == highest]
    if len(modes) != 1:
        raise ValueError("No unique common reference")
    master = next(p for row in rows for p in row if _key(p) == modes[0])
    payload = set(map(int, np.unique(master)))
    if len(payload) != 2 or frame in payload:
        raise ValueError("Reference must have two non-frame colors")
    patches, columns, boxes, edits = [], [], [], []
    for row in rows:
        if any(not set(map(int, np.unique(p))) <= payload for p in row):
            raise ValueError("Unexpected library color")
        odd = [j for j, p in enumerate(row) if not np.array_equal(p, master)]
        if len(odd) != 1:
            raise ValueError("Each band needs one visible exception")
        column = odd[0]
        variant = row[column]
        changed = np.argwhere(master != variant)
        lo, hi = changed.min(axis=0), changed.max(axis=0) + 1
        r, c, h, w = int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1])
        patches.append((master[r:r+h, c:c+w].copy(), variant[r:r+h, c:c+w].copy()))
        boxes.append((r, c, h, w))
        columns.append(column)
        edits.append(len(changed))
    return dict(lib=lib, frame=frame, ph=ph, pw=pw, bands=bands,
                copies=copies, master=master, rows=rows, payload=payload,
                patches=patches, columns=tuple(columns), boxes=tuple(boxes),
                edits=tuple(edits))


def _orient(array, side):
    # side denotes the SOURCE side; the identical map applies to its destination.
    if side == "left":
        return array.copy()
    if side == "right":
        return array[:, ::-1].copy()
    if side == "top":
        return array.T.copy()
    return array.T[::-1].copy()


def _layouts(grid):
    normalizations = (("left", grid), ("right", grid[:, ::-1]),
                      ("top", grid.T), ("bottom", grid[::-1].T))
    for side, g in normalizations:
        for frame in map(int, np.unique(g)):
            full = np.all(g == frame, axis=0)
            for split in np.flatnonzero(full[:-1] & full[1:]):
                split = int(split)
                lib, board = g[:, :split+1], g[:, split+2:]
                if board.shape[1] < 3 or np.any(board == frame):
                    continue
                try:
                    info = _library(lib, frame)
                except ValueError:
                    continue
                background = int(board[0, 0])
                if (background in info["payload"]
                        or not np.all(board[[0, -1]] == background)
                        or not np.all(board[:, [0, -1]] == background)):
                    continue
                other = set(map(int, np.unique(board))) - info["payload"] - {background}
                if len(other) != 1:
                    continue
                vacancy = other.pop()
                clues = (board != background) & (board != vacancy)
                mask = board != background
                if not clues.any() or not (board == vacancy).any():
                    continue
                if sum(before.size for before, after in info["patches"]) != int(mask.sum()):
                    continue
                info.update(side=side, normalized=g.copy(), board=board.copy(),
                            background=background, vacancy=vacancy,
                            mask=mask, clues=clues)
                yield info


class _SearchLimit(ValueError):
    pass


def _cover(board, background, vacancy, patches, *, match="before", max_nodes=None):
    """Enumerate at most two rendered answers with finite before/after inventory.

    Matching and painting are different operations. Identical records are grouped
    with counts. A search limit raises instead of pretending to prove uniqueness.
    """
    mask = board != background
    clues = mask & (board != vacancy)
    coords = np.argwhere(mask)
    if not len(coords) or sum(a.size for a, b in patches) != len(coords):
        return {}, [], 0
    ids = np.full(board.shape, -1, dtype=int)
    for i, (r, c) in enumerate(coords):
        ids[r, c] = i
    groups, lookup = [], {}
    for before, after in patches:
        key = (_key(before), _key(after))
        if key in lookup:
            groups[lookup[key]]["count"] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(before=before, after=after, count=1))
    placements, by_cell = [], [[] for _ in coords]
    H, W = board.shape
    for group, item in enumerate(groups):
        before, after = item["before"], item["after"]
        h, w = before.shape
        pattern = before if match == "before" else after
        for r in range(H-h+1):
            for c in range(W-w+1):
                if not np.all(mask[r:r+h, c:c+w]):
                    continue
                visible = clues[r:r+h, c:c+w]
                if match is not None and np.any(board[r:r+h, c:c+w][visible] != pattern[visible]):
                    continue
                cells = tuple(map(int, ids[r:r+h, c:c+w].ravel()))
                bits = sum(1 << i for i in cells)
                assignments = tuple(zip(cells, map(int, after.ravel())))
                p = len(placements)
                placements.append((group, r, c, bits, assignments))
                for i in cells:
                    by_cell[i].append(p)
    remaining = [item["count"] for item in groups]
    full = (1 << len(coords)) - 1
    painted = [-1] * len(coords)
    outcomes, path, dead = {}, [], set()
    nodes = 0

    def search(covered):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise _SearchLimit("Exact-cover construction budget exceeded")
        if covered == full:
            if not any(remaining):
                outcomes.setdefault(tuple(painted), tuple(path))
                return True
            return False
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            options = [p for p in by_cell[i] if remaining[placements[p][0]]
                       and not covered & placements[p][3]]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for p in best:
            group, r, c, bits, assignments = placements[p]
            remaining[group] -= 1
            path.append((group, r, c))
            for i, color in assignments:
                painted[i] = color
            found |= search(covered | bits)
            path.pop()
            remaining[group] += 1
            for i, color in assignments:
                painted[i] = -1
            if len(outcomes) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    search(0)
    return outcomes, groups, nodes


def _analyze_input(input_grid):
    grid = _grid(input_grid)
    answers = []
    for info in _layouts(grid):
        outcomes, groups, nodes = _cover(info["board"], info["background"],
                                        info["vacancy"], info["patches"])
        if not outcomes:
            continue
        if len(outcomes) > 1:
            raise ValueError("More than one rendered mosaic")
        rendering, path = next(iter(outcomes.items()))
        out = info["board"].copy()
        out[info["mask"]] = rendering
        info.update(output=_orient(out, info["side"]), canonical_output=out,
                    groups=groups, path=path, nodes=nodes)
        answers.append(info)
    if len(answers) != 1:
        raise ValueError("Expected one visible library/destination interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict solely from the visible input, without seeds or stored examples."""
    return _analyze_input(input_grid)["output"]


def _assemble(master, variants, columns, copies, frame):
    ph, pw = master.shape
    lib = np.full((len(variants)*(ph+1)+1, copies*(pw+1)+1), frame, dtype=np.int8)
    for band, variant in enumerate(variants):
        for col in range(copies):
            r, c = 1+band*(ph+1), 1+col*(pw+1)
            lib[r:r+ph, c:c+pw] = variant if col == columns[band] else master
    return lib


def _reference(rng, ph, pw, p, q):
    rr, cc = np.indices((ph, pw))
    style = int(rng.integers(6))
    if style == 0:
        bits = (rr+cc) % 2 == int(rng.integers(2))
    elif style == 1:
        bits = (rr if rng.random() < .5 else cc) % int(rng.integers(2, 4)) == 0
    elif style == 2:
        bits = (rr == 0) | (rr == ph-1) | (cc == 0) | (cc == pw-1)
    elif style == 3:
        bits = (rr-cc) % int(rng.integers(2, 5)) == 0
    elif style == 4:
        bits = rng.random((ph, pw)) < float(rng.uniform(.2, .8))
    else:
        bits = rng.random((ph, pw)) < .35
        bits |= bits[::-1]
    bits[0, 0], bits[-1, -1] = True, False
    return np.where(bits, p, q).astype(np.int8)


def _record(rng, master, p, q, box=None, balanced=False):
    ph, pw = master.shape
    if box is None:
        choices = [(h, w) for h in range(1, ph+1) for w in range(1, pw+1) if h*w >= 2]
        h, w = choices[int(rng.integers(len(choices)))]
        r, c = int(rng.integers(ph-h+1)), int(rng.integers(pw-w+1))
    else:
        r, c, h, w = box
    before = master[r:r+h, c:c+w].copy()
    flips = np.zeros((h, w), dtype=bool)
    if not balanced:
        flips = rng.random((h, w)) < float(rng.uniform(.15, .70))
    flips[0, 0] = flips[-1, -1] = True
    after = before.copy()
    after[flips] = np.where(before[flips] == p, q, p)
    variant = master.copy()
    variant[r:r+h, c:c+w] = after
    return before, after, variant


def _grow(rng, shapes, H, W, connected):
    mask = np.zeros((H, W), dtype=bool)
    rects = []
    for h, w in shapes:
        legal = set()
        for _ in range(140):
            if rects and (connected or rng.random() < .78):
                y, x, th, tw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    r, c = (y-h if side == 0 else y+th), x+int(rng.integers(1-w, tw))
                else:
                    r, c = y+int(rng.integers(1-h, th)), (x-w if side == 2 else x+tw)
            else:
                if H-h-1 < 1 or W-w-1 < 1:
                    return None
                r, c = int(rng.integers(1, H-h)), int(rng.integers(1, W-w))
            if (r < 1 or c < 1 or r+h >= H or c+w >= W
                    or mask[r:r+h, c:c+w].any()):
                continue
            legal.add((r, c))
        if not legal:
            return None
        legal = sorted(legal)  # never depend on randomized set iteration
        r, c = legal[int(rng.integers(len(legal)))]
        rects.append((r, c, h, w))
        mask[r:r+h, c:c+w] = True
    return mask, rects


def _paint(rng, shape, rects, records, roles, witness=False):
    frame, background, vacancy, p, q = roles
    board = np.full(shape, background, dtype=np.int8)
    out = board.copy()
    for i, ((r, c, h, w), (before, after, variant)) in enumerate(zip(rects, records)):
        changed = np.flatnonzero(before != after)
        unchanged = np.flatnonzero(before == after)
        count = min(before.size-1, max(1, int(round(before.size*float(rng.uniform(.32, .67))))))
        chosen = {int(rng.choice(changed))}
        if len(unchanged):
            chosen.add(int(rng.choice(unchanged)))
        count = max(count, len(chosen))
        if witness and i < 2:
            count = 2
        eligible = [j for j in range(before.size) if j not in chosen]
        if count > len(chosen):
            chosen.update(map(int, rng.choice(eligible, count-len(chosen), replace=False)))
        tile = np.full(before.shape, vacancy, dtype=np.int8)
        tile.flat[list(chosen)] = before.flat[list(chosen)]
        board[r:r+h, c:c+w] = tile
        out[r:r+h, c:c+w] = after
    return board, out


def _join(lib, board, out, frame, side):
    divider = np.full((lib.shape[0], 1), frame, dtype=np.int8)
    return {"input": _orient(np.concatenate((lib, divider, board), axis=1), side),
            "output": _orient(out, side)}


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in cells:
                    cells.remove(point)
                    stack.append(point)
                    group.add(point)
        result.append(group)
    return result


def _features(info):
    owner = np.full(info["board"].shape, -1, dtype=int)
    hidden_unchanged = changed_clues = 0
    for i, (group, r, c) in enumerate(info["path"]):
        before, after = info["groups"][group]["before"], info["groups"][group]["after"]
        h, w = before.shape
        owner[r:r+h, c:c+w] = i
        visible = info["clues"][r:r+h, c:c+w]
        hidden_unchanged += int(np.count_nonzero(~visible & (before == after)))
        changed_clues += int(np.count_nonzero(visible & (before != after)))
    components = _components(info["mask"])
    max_join = max(len({int(owner[r, c]) for r, c in group}) for group in components)
    patches = info["patches"]
    balanced = any(Counter(map(int, a.ravel())) == Counter(map(int, b.ravel())) for a, b in patches)
    differing_same_size = any(a.shape == b.shape and not np.array_equal(a, b)
                              for i, (_, a) in enumerate(patches) for _, b in patches[i+1:])
    repeated = len({(_key(a), _key(b)) for a, b in patches}) < len(patches)
    return dict(side=info["side"], bands=info["bands"], ph=info["ph"], pw=info["pw"],
                copies=info["copies"], shape=info["board"].shape,
                patch_shapes=tuple(a.shape for a, b in patches),
                columns=info["columns"], balanced=balanced,
                same_size_different=differing_same_size, repeated=repeated,
                hidden_unchanged=hidden_unchanged, changed_clues=changed_clues,
                components=len(components), max_join=max_join,
                coverage=float(info["mask"].mean()), clue_density=float(info["clues"].sum()/info["mask"].sum()),
                edits=info["edits"], master=_key(info["master"] == info["master"][0, 0]),
                roles=(info["frame"], info["background"], info["vacancy"], *sorted(info["payload"])))


def _witness(info):
    f = _features(info)
    if (not f["balanced"] or not f["same_size_different"] or f["hidden_unchanged"] < 1
            or len(set(f["columns"])) < 2 or f["max_join"] < 2
            or not any(a.shape != info["master"].shape for a, b in info["patches"])):
        return False
    outcomes, _, _ = _cover(info["board"], info["background"], info["vacancy"],
                            info["patches"], match=None, max_nodes=10000)
    return len(outcomes) >= 2 and _excluded_local_fit(info)


def _excluded_local_fit(info):
    """Find a before-compatible translation that cannot join any complete cover.

    A locally fitting pattern is insufficient: it must leave a tileable remainder.
    Pinning a candidate is checked by removing its rectangle and one record from
    the remaining exact-cover problem, without consulting the output label.
    """
    board = info["board"]
    H, W = board.shape
    chosen = {(group, r, c) for group, r, c in info["path"]}
    for group, item in enumerate(info["groups"]):
        before, after = item["before"], item["after"]
        h, w = before.shape
        for r in range(H-h+1):
            for c in range(W-w+1):
                if (group, r, c) in chosen or not info["mask"][r:r+h, c:c+w].all():
                    continue
                visible = info["clues"][r:r+h, c:c+w]
                if np.any(board[r:r+h, c:c+w][visible] != before[visible]):
                    continue
                reduced = board.copy()
                reduced[r:r+h, c:c+w] = info["background"]
                remaining = list(info["patches"])
                key = (_key(before), _key(after))
                index = next(i for i, (a, b) in enumerate(remaining) if (_key(a), _key(b)) == key)
                remaining.pop(index)
                covers, _, _ = _cover(reduced, info["background"], info["vacancy"],
                                      remaining, max_nodes=5000)
                if not covers:
                    return True
    return False


def _fallback(rng, side, roles, bands, copies):
    # Alternating two-row steps: each 2x2 block has a forced footprint, while its
    # first-column before clues distinguish two same-size after patterns.
    frame, background, vacancy, p, q = roles
    ph, pw = 4, 3
    master = _reference(rng, ph, pw, p, q)
    master[:2, :2] = [[p, p], [q, q]]
    master[2:, 1:] = [[q, q], [p, p]]
    types = ([0, 0, 1] + list(map(int, rng.integers(2, size=bands-3)))) if bands >= 3 else [0, 1]
    records = [_record(rng, master, p, q, (0, 0, 2, 2) if t == 0 else (2, 1, 2, 2), True)
               for t in types]
    H, W = bands*5+1, bands*2+2
    base = int(rng.integers(1, H-3))
    rects = [(base+(0 if i < 2 else (i-1) % 2), 1+2*i, 2, 2) for i in range(bands)]
    board = np.full((H, W), background, dtype=np.int8)
    out = board.copy()
    for (r, c, h, w), (before, after, _) in zip(rects, records):
        board[r:r+2, c:c+2] = vacancy
        board[r:r+2, c] = before[:, 0]
        out[r:r+2, c:c+2] = after
    order = list(map(int, rng.permutation(bands)))
    variants = [records[i][2] for i in order]
    columns = list(map(int, rng.integers(copies, size=bands)))
    columns[1] = (columns[0]+1) % copies
    lib = _assemble(master, variants, columns, copies, frame)
    pair = _join(lib, board, out, frame, side)
    info = _analyze_input(pair["input"])
    if not np.array_equal(info["output"], pair["output"]) or (bands >= 3 and not _witness(info)):
        raise AssertionError("Fallback construction violated the rule")
    return pair


def _make_pair(rng, side, roles, testing=False, witness=False):
    frame, background, vacancy, p, q = roles
    bands = int(rng.integers(4, 6)) if testing else (3 if witness else int(rng.integers(2, 4)))
    copies = int(rng.integers(3, 5))
    for attempt in range(100):
        max_ph = min(6, 29//bands-1)
        ph = int(rng.integers(4 if witness else 3, max_ph+1))
        pw = int(rng.integers(3, 5 if copies == 4 else 6))
        H, source_width = bands*(ph+1)+1, copies*(pw+1)+1
        max_width = 30-source_width-1
        if max_width < pw+2:
            continue
        W = int(rng.integers(max(pw+2, 6), max_width+1))
        master = _reference(rng, ph, pw, p, q)
        records = []
        if witness:
            master[:2, :2] = [[p, p], [q, q]]
            master[-2:, -2:] = [[q, q], [p, p]]
            first = _record(rng, master, p, q, (0, 0, 2, 2), True)
            records = [first, tuple(x.copy() for x in first),
                       _record(rng, master, p, q, (ph-2, pw-2, 2, 2), True)]
        while len(records) < bands:
            if records and rng.random() < .24:
                records.append(tuple(x.copy() for x in records[int(rng.integers(len(records)))]))
            else:
                records.append(_record(rng, master, p, q))
        if witness:
            # Two identical before records form a broad 2x4 join. Their one-cell
            # shifted local match straddles the hidden seam but leaves untileable
            # boundary strips. A distinct same-size record joins above or below.
            r, c = int(rng.integers(3, H-4)), int(rng.integers(1, W-4))
            third_row = r + int(rng.choice([-2, 2]))
            rects = [(r, c, 2, 2), (r, c+2, 2, 2),
                     (third_row, c+int(rng.integers(3)), 2, 2)]
            mask = np.zeros((H, W), dtype=bool)
            for y, x, h, w in rects:
                mask[y:y+h, x:x+w] = True
            grown = mask, rects
        else:
            grown = _grow(rng, [a.shape for a, b, v in records], H, W,
                          connected=testing or rng.random() < .70)
        if grown is None:
            continue
        mask, rects = grown
        if len(_components(mask)) == bands:
            continue
        board, out = _paint(rng, (H, W), rects, records, roles, witness)
        try:
            outcomes, _, _ = _cover(board, background, vacancy, [(a, b) for a, b, v in records], max_nodes=5000)
            expected = tuple(map(int, out[mask]))
            if len(outcomes) != 1 or expected not in outcomes:
                continue
            order = list(map(int, rng.permutation(bands)))
            columns = list(map(int, rng.integers(copies, size=bands)))
            if witness and len(set(columns)) == 1:
                columns[1] = (columns[0]+1) % copies
            lib = _assemble(master, [records[i][2] for i in order], columns, copies, frame)
            pair = _join(lib, board, out, frame, side)
            info = _analyze_input(pair["input"])
            if np.array_equal(info["output"], out if side == "left" else _orient(out, side)):
                if not witness or _witness(info):
                    return pair
        except _SearchLimit:
            continue
    return _fallback(rng, side, roles, bands, copies)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    horizontal = str(rng.choice(["left", "right"]))
    vertical = str(rng.choice(["top", "bottom"]))
    train_sides = [horizontal, vertical]
    rng.shuffle(train_sides)
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    puzzle, seen, previous_roles = {"train": [], "test": []}, set(), None
    for i in range(num_train+num_test):
        testing = i >= num_train
        side = opposite[train_sides[(i-num_train) % 2]] if testing else train_sides[i % 2]
        for retry in range(16):
            roles = tuple(map(int, rng.choice(10, 5, replace=False)))
            if (previous_roles is not None and roles[:3] == previous_roles[:3]
                    and set(roles[3:]) == set(previous_roles[3:])):
                roles = roles[1:] + roles[:1]
            pair = _make_pair(rng, side, roles, testing=testing, witness=i == 0)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("Unique input construction exhausted")
        previous_roles = roles
        seen.add(_key(pair["input"]))
        puzzle["test" if testing else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("Puzzle requires exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise AssertionError("Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Repeated input"
            seen.add(_key(inp))
            assert len(np.unique(inp)) == 5
            info = _analyze_input(inp)
            assert np.array_equal(info["output"], out), "Wrong rendered mosaic"
            assert len(info["path"]) == info["bands"], "Inventory not used exactly once"
            assert np.all(info["canonical_output"][~info["mask"]] == info["background"])
            for group, r, c in info["path"]:
                before, after = info["groups"][group]["before"], info["groups"][group]["after"]
                h, w = before.shape
                visible = info["clues"][r:r+h, c:c+w]
                target = info["board"][r:r+h, c:c+w]
                assert np.any(target == info["vacancy"]), "Every generated patch has a vacancy"
                assert np.any(visible & (before != after)), "Demonstrate overwriting an old clue"
                assert np.all(target[visible] == before[visible])
                assert np.array_equal(info["canonical_output"][r:r+h, c:c+w], after)
                if np.any(before == after):
                    assert np.any(visible & (before == after)), "Show unchanged crop correspondence"
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    assert _witness(train[0]), "Missing distinguishing registration demonstration"
    assert train[0]["bands"] == 3
    assert all(a["bands"] <= 3 for a in train)
    assert all(a["bands"] >= 4 for a in test)
    assert {a["side"] in ("left", "right") for a in train[:2]} == {True, False}
    assert {a["side"] for a in train}.isdisjoint({a["side"] for a in test})
    assert train[0]["frame"] != train[1]["frame"] or train[0]["payload"] != train[1]["payload"] or train[0]["background"] != train[1]["background"] or train[0]["vacancy"] != train[1]["vacancy"]
    return None


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in puzzle[s])


def self_test():
    """Bounded 42-seed coverage plus semantic and metamorphic checks."""
    puzzles, features = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        features.extend(_features(_analyze_input(p["input"])) for s in ("train", "test") for p in puzzle[s])
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(17)) == _signature(puzzles[17])
    assert {f["side"] for f in features} == {"left", "right", "top", "bottom"}
    assert {f["bands"] for f in features} == {2, 3, 4, 5}
    assert {f["copies"] for f in features} == {3, 4}
    assert len({(f["ph"], f["pw"]) for f in features}) >= 8
    assert len({f["shape"] for f in features}) >= 30
    assert len({f["master"] for f in features}) >= 80
    assert len({f["patch_shapes"] for f in features}) >= 60
    assert any(f["repeated"] for f in features)
    assert any(f["components"] > 1 for f in features)
    assert max(f["max_join"] for f in features) == 5
    assert max(max(f["edits"]) for f in features) >= 10
    assert {f["roles"][0] for f in features} == set(range(10))
    assert max(f["clue_density"] for f in features) - min(f["clue_density"] for f in features) > .15
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(12000+nt*10+ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    base = puzzles[0]["train"][0]
    info = _analyze_input(base["input"])
    # Replay before instead of after: removing A's actual anomaly changes pixels.
    old = info["board"].copy()
    for group, r, c in info["path"]:
        before = info["groups"][group]["before"]
        h, w = before.shape
        old[r:r+h, c:c+w] = before
    assert not np.array_equal(old, info["canonical_output"])
    # The target matches old pixels, never the changed clues of the correct after.
    after_matches, _, _ = _cover(info["board"], info["background"], info["vacancy"],
                                info["patches"], match="after", max_nodes=10000)
    assert tuple(map(int, info["canonical_output"][info["mask"]])) not in after_matches
    # A lattice permutation cannot be used as a destination-order shortcut.
    variants = [row[col] for row, col in zip(info["rows"], info["columns"])]
    lib = _assemble(info["master"], variants[::-1],
                    [(col+1) % info["copies"] for col in info["columns"][::-1]],
                    info["copies"], info["frame"])
    permuted = _join(lib, info["board"], info["canonical_output"], info["frame"], info["side"])
    assert np.array_equal(solve(permuted["input"]), base["output"])
    # Moving the stencil while retaining the library changes the destination answer.
    # Shift the complete active region, preserving its local pattern orientation.
    cells = np.argwhere(info["mask"])
    lo, hi = int(cells[:, 0].min()), int(cells[:, 0].max())
    delta = -1 if lo > 1 else 1 if hi < info["board"].shape[0]-2 else 0
    if delta:
        board = np.full_like(info["board"], info["background"])
        out = board.copy()
        for r, c in cells:
            board[r+delta, c] = info["board"][r, c]
            out[r+delta, c] = info["canonical_output"][r, c]
        shifted = _join(info["lib"], board, out, info["frame"], info["side"])
        assert np.array_equal(solve(shifted["input"]), shifted["output"])
        assert not np.array_equal(shifted["output"], base["output"])
    # Finite inventory and rendered, rather than identity, uniqueness.
    board = np.array([[0]*6, [0, 4, 4, 4, 4, 0], [0]*6], dtype=np.int8)
    before, after = np.array([[1, 2]]), np.array([[2, 1]])
    assert not _cover(board, 0, 4, [(before, after)])[0]
    assert len(_cover(board, 0, 4, [(before, after)]*2)[0]) == 1
    other = (np.array([[2, 1]]), np.array([[1, 2]]))
    assert len(_cover(board, 0, 4, [(before, after), other])[0]) == 2
    # Explicitly exercise the fallback on all sides and both inventory regimes.
    for side in ("left", "right", "top", "bottom"):
        for bands in (3, 5):
            pair = _fallback(np.random.default_rng(90), side, (0, 1, 2, 3, 4), bands, 4)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    # Corruption checks: no silent repair of a missing exception or wrong output.
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["train"][0]["output"][1, 1] = (int(bad["train"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Wrong output accepted")
    missing = _assemble(info["master"], [info["master"]]+variants[1:],
                        info["columns"], info["copies"], info["frame"])
    broken = _join(missing, info["board"], info["canonical_output"], info["frame"], info["side"])
    try:
        solve(broken["input"])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing exception accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_board_shapes": len({f["shape"] for f in features}),
            "distinct_patch_shape_sequences": len({f["patch_shapes"] for f in features}),
            "distinct_normalized_master_bitmaps": len({f["master"] for f in features})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
