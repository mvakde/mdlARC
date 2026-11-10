"""Revision plates: infer signed changes, then print them through a color key.

No parent imports, files, latent generation state, or training labels are used by
solve. Generation writes nothing. All diagnostics returned here are compactable
summary values, not retained puzzle payloads.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    p = np.asarray(cells)
    return int(p[:, 0].min()), int(p[:, 0].max()) + 1, int(p[:, 1].min()), int(p[:, 1].max()) + 1


def _slice(a, box):
    r, e, c, d = box
    return a[r:e, c:d]


def _frame(a, color):
    a[0, :] = a[-1, :] = color
    a[:, 0] = a[:, -1] = color


def _signed_asymmetric(plus, minus):
    signed = plus.astype(np.int8) - minus.astype(np.int8)
    transforms = [np.rot90(signed, k) for k in (1, 2, 3)]
    transforms += [np.rot90(signed[:, ::-1], k) for k in range(4)]
    return all(not np.array_equal(signed, t) for t in transforms)


def _parse(input_grid):
    a = _grid(input_grid)
    frames, catalogues = [], []
    for color in map(int, np.unique(a)):
        cells = np.argwhere(a == color)
        box = _box(cells)
        tile = _slice(a, box)
        h, w = tile.shape
        if min(h, w) < 4:
            continue
        mask = tile == color
        rs = np.flatnonzero(np.all(mask, axis=1))
        cs = np.flatnonzero(np.all(mask, axis=0))
        if not (len(rs) >= 2 and len(cs) >= 2 and rs[0] == cs[0] == 0
                and rs[-1] == h - 1 and cs[-1] == w - 1):
            continue
        expected = np.zeros(tile.shape, dtype=bool)
        expected[rs, :] = True
        expected[:, cs] = True
        if not np.array_equal(mask, expected):
            continue
        if len(rs) == len(cs) == 2:
            if min(h, w) >= 6:
                frames.append((color, box))
        elif (max(len(rs), len(cs)) >= 4 and np.all(np.diff(rs) == rs[1])
              and np.all(np.diff(cs) == cs[1]) and rs[1] == cs[1]
              and 2 <= rs[1] - 1 <= 5):
            catalogues.append((color, box, rs, cs))
    _require(len(frames) == len(catalogues) == 1, "need one frame and one catalogue")
    frame_color, frame_box = frames[0]
    lattice, cat_box, rs, cs = catalogues[0]
    r, e, c, d = frame_box
    corners = [a[r + 1, c + 1], a[r + 1, d - 2], a[e - 2, c + 1], a[e - 2, d - 2]]
    _require(len(set(map(int, corners))) == 1, "empty inner corners must identify background")
    bg = int(corners[0])
    _require(len({bg, frame_color, lattice}) == 3, "structural colors must differ")
    _require(e <= cat_box[0] or cat_box[1] <= r or d <= cat_box[2] or cat_box[3] <= c,
             "catalogue overlaps frame")
    n = int(rs[1] - 1)
    groups, source_rows = {}, []
    cr, _, cc, _ = cat_box
    for y in rs[:-1]:
        source_row = []
        for x in cs[:-1]:
            box = (cr + int(y) + 1, cr + int(y) + n + 1,
                   cc + int(x) + 1, cc + int(x) + n + 1)
            panel = _slice(a, box)
            colors = set(map(int, np.unique(panel))) - {bg}
            _require(len(colors) == 1, "panel must use exactly one source")
            source = next(iter(colors))
            _require(source not in (frame_color, lattice), "source collides with structure")
            bitmap = panel == source
            _require(0 < bitmap.sum() < n * n, "panel must be nonempty and nonsolid")
            groups.setdefault(source, []).append((bitmap, box))
            source_row.append(source)
        source_rows.append(source_row)
    _require(1 <= len(groups) <= 4, "invalid source count")
    entries = {}
    for source, panels in groups.items():
        _require(3 <= len(panels) <= 6, "invalid copy count")
        counts = Counter(p.tobytes() for p, _ in panels)
        _require(len(counts) == 2 and sorted(counts.values()) == [1, len(panels) - 1],
                 "need exactly one visible revision against a majority")
        master_key = counts.most_common(1)[0][0]
        master = next(p for p, _ in panels if p.tobytes() == master_key)
        index = next(i for i, (p, _) in enumerate(panels) if p.tobytes() != master_key)
        odd = panels[index][0]
        plus, minus = odd & ~master, master & ~odd
        _require(1 <= int(plus.sum() + minus.sum()) <= n * n // 2, "invalid edit count")
        entries[source] = dict(master=master, odd=odd, plus=plus, minus=minus,
                               index=index, panels=panels, copies=len(panels))
    first = next(iter(entries.values()))
    _require(all(np.array_equal(t["master"], first["master"]) for t in entries.values()),
             "catalogue needs a common binary master")
    _require(len({t["copies"] for t in entries.values()}) == 1, "inconsistent copies")
    # Each source occupies one complete catalogue band, including after rotation.
    sources_by_cell = np.array(source_rows)
    row_bands = (sources_by_cell.shape[0] == len(entries)
                 and all(len(set(map(int, row))) == 1 for row in sources_by_cell))
    col_bands = (sources_by_cell.shape[1] == len(entries)
                 and all(len(set(map(int, col))) == 1 for col in sources_by_cell.T))
    _require(row_bands or col_bands, "sources must form catalogue bands")
    interior = a[r + 1:e - 1, c + 1:d - 1]
    _require(np.all(interior[[0, -1], :] == bg) and np.all(interior[:, [0, -1]] == bg),
             "sheet needs a background moat")
    places = []
    for cells in _components(interior != bg):
        box = _box(cells)
        block = _slice(interior, box)
        source = int(block[0, 0])
        _require(block.shape == (n, n) and len(cells) == n * n
                 and np.all(block == source) and source in entries, "invalid sheet square")
        places.append((box[0] + r + 1, box[2] + c + 1, source))
    _require({s for _, _, s in places} == set(entries), "every source must occur on sheet")
    outside = a != bg
    _slice(outside, frame_box)[:] = False
    _slice(outside, cat_box)[:] = False
    mapping, keys = {}, {}
    for cells in _components(outside):
        box = _box(cells)
        tile = _slice(a, box)
        _require(len(cells) == tile.size and len(np.unique(tile)) == 2, "invalid key rectangle")
        candidates = set()
        for q in range(4):
            mask = np.rot90(_ARROW, q)
            if mask.shape == tile.shape:
                source, target = int(tile[mask][0]), int(tile[~mask][0])
                if source != target and np.all(tile[mask] == source) and np.all(tile[~mask] == target):
                    candidates.add((source, target))
        _require(len(candidates) == 1, "unknown arrow")
        source, target = next(iter(candidates))
        _require(source in entries and source not in mapping, "invalid arrow source")
        _require(target not in (bg, frame_color, lattice), "invalid target role")
        mapping[source], keys[source] = target, box
    _require(set(mapping) == set(entries), "missing arrow")
    side = ("above" if cat_box[1] <= r else "below" if cat_box[0] >= e
            else "left" if cat_box[3] <= c else "right")
    return dict(grid=a, bg=bg, frame=frame_color, lattice=lattice, frame_box=frame_box,
                cat_box=cat_box, n=n, entries=entries, mapping=mapping, keys=keys,
                places=places, side=side, copies=first["copies"])


def _render(info, mode="signed"):
    a, bg, n = info["grid"], info["bg"], info["n"]
    r, _, c, _ = info["frame_box"]
    result = _slice(a, info["frame_box"]).copy()
    for y, x, source in info["places"]:
        t = info["entries"][source]
        target = info["mapping"][source]
        if mode == "recursive":
            visited = {source}
            while target in info["mapping"] and target not in visited:
                visited.add(target)
                target = info["mapping"][target]
        block = result[y - r:y - r + n, x - c:x - c + n]
        block[:] = bg
        if mode == "odd":
            block[t["odd"]] = target
        elif mode == "no_mapping":
            block[t["plus"] | t["minus"]] = source
        elif mode == "unsigned":
            block[t["plus"] | t["minus"]] = target
        else:
            block[t["minus"]] = source
            block[t["plus"]] = target
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the signed-print output entirely from the visible input."""
    return _render(_parse(input_grid))


def _master(rng, n):
    y, x = np.indices((n, n))
    style = int(rng.integers(7))
    if style == 0:
        mask = rng.random((n, n)) < rng.uniform(.2, .8)
    elif style == 1:
        mask = (x == int(rng.integers(n))) | (y == int(rng.integers(n)))
    elif style == 2:
        mask = (x == 0) | (y == 0) | (x == n - 1) | (y == n - 1)
    elif style == 3:
        mask = (x + y) % 2 == int(rng.integers(2))
    elif style == 4:
        mask = (x == y) | ((x + y == n - 1) & (rng.random((n, n)) < .5))
    elif style == 5:
        mask = (y if rng.random() < .5 else x) % 2 == 0
    else:
        mask = np.zeros((n, n), dtype=bool)
        mask[int(rng.integers(n)), int(rng.integers(n))] = True
        for _ in range(int(rng.integers(1, n * n))):
            border = [(r, c) for r in range(n) for c in range(n) if not mask[r, c]
                      and any(0 <= a < n and 0 <= b < n and mask[a, b]
                              for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
            if border:
                mask[border[int(rng.integers(len(border)))]] = True
    if rng.random() < .2:
        mask |= mask[:, ::-1]
    if rng.random() < .25:
        mask = ~mask
    # Reserve enough old ink and blank cells for informative mixed revisions.
    low = 2 if n == 2 else 3
    if mask.sum() < low:
        blanks = np.flatnonzero(~mask)
        mask.flat[rng.choice(blanks, low - int(mask.sum()), replace=False)] = True
    if mask.sum() > n * n - low:
        ink = np.flatnonzero(mask)
        mask.flat[rng.choice(ink, int(mask.sum()) - (n*n-low), replace=False)] = False
    return mask


def _edit(rng, master, kind=None, asymmetric=False):
    n = master.shape[0]
    ink, blank = np.flatnonzero(master), np.flatnonzero(~master)
    kind = kind or str(rng.choice(["add", "remove", "balanced", "mixed"]))
    cap = n * n // 2
    plus = np.zeros(master.shape, dtype=bool)
    minus = plus.copy()
    if asymmetric:
        # A colored ordered pair of offsets can distinguish all eight symmetries.
        candidates = []
        for p in blank:
            for m in ink:
                plus.flat[p], minus.flat[m] = True, True
                if _signed_asymmetric(plus, minus):
                    candidates.append((int(p), int(m)))
                plus.flat[p], minus.flat[m] = False, False
        _require(bool(candidates), "no asymmetric balanced witness")
        p, m = candidates[int(rng.integers(len(candidates)))]
        plus.flat[p], minus.flat[m] = True, True
    else:
        if kind == "add":
            add, remove = int(rng.integers(1, min(cap, len(blank)-1)+1)), 0
        elif kind == "remove":
            add, remove = 0, int(rng.integers(1, min(cap, len(ink)-1)+1))
        elif kind == "balanced":
            amount = int(rng.integers(1, min(cap//2, len(ink)-1, len(blank)-1)+1))
            add = remove = amount
        else:
            choices = [(p, m) for p in range(1, min(cap, len(blank)-1)+1)
                       for m in range(1, min(cap, len(ink)-1)+1) if p+m <= cap]
            unbalanced = [(p, m) for p, m in choices if p != m]
            add, remove = (unbalanced or choices)[int(rng.integers(len(unbalanced or choices)))]
        if add:
            plus.flat[rng.choice(blank, add, replace=False)] = True
        if remove:
            minus.flat[rng.choice(ink, remove, replace=False)] = True
    odd = (master | plus) & ~minus
    return odd, plus, minus


def _mapping(rng, sources, external, kind):
    order = list(map(int, rng.permutation(sources)))
    k = len(sources)
    if k == 1 or kind == "disjoint":
        targets = list(map(int, rng.choice(external, k, replace=k > len(external))))
        return dict(zip(sources, targets))
    if kind == "cycle":
        return {s: order[(i+1) % k] for i, s in enumerate(order)}
    if kind == "chain":
        return {s: order[i+1] if i+1 < k else int(rng.choice(external)) for i, s in enumerate(order)}
    if kind == "many":
        target = int(rng.choice(external))
        return {s: target for s in sources}
    choices = sources + external
    return {s: int(rng.choice([t for t in choices if t != s])) for s in sources}


def _starts(rng, length, number, n):
    slack = length - (number * (n + 1) + 3)
    gaps = rng.multinomial(slack, np.full(number + 1, 1 / (number + 1)))
    result, position = [], 2 + int(gaps[0])
    for i in range(number):
        result.append(position)
        position += n + 1 + int(gaps[i + 1])
    return result


def _make_pair(rng, n, k, copies, q, witness=0):
    palette = list(map(int, rng.permutation(10)))
    bg, frame_color, lattice = palette[:3]
    sources, external = palette[3:3+k], palette[3+k:]
    mapping = _mapping(rng, sources, external, "chain" if witness == 1 else
                       str(rng.choice(["disjoint", "chain", "cycle", "many", "mixed"])))
    master = _master(rng, n)
    rows, masks = [], {}
    columns = list(map(int, rng.integers(copies, size=k)))
    if witness == 1:
        columns[1] = (columns[0] + int(rng.integers(1, copies))) % copies
    for i, source in enumerate(sources):
        kind = ("balanced" if i == 0 else "mixed") if witness == 1 else (
            ("add" if i == 0 else "remove") if witness == 2 else None)
        odd, plus, minus = _edit(rng, master, kind, asymmetric=witness == 1 and i == 0)
        panels = [master for _ in range(copies)]
        panels[columns[i]] = odd
        rows.append([np.where(p, source, bg).astype(np.int8) for p in panels])
        masks[source] = (plus, minus)
    ch, cw = k * (n + 1) + 1, copies * (n + 1) + 1
    catalogue = np.full((ch, cw), lattice, dtype=np.int8)
    for i, row in enumerate(rows):
        for j, panel in enumerate(row):
            catalogue[1+i*(n+1):1+i*(n+1)+n, 1+j*(n+1):1+j*(n+1)+n] = panel
    bank_h, bank_w = max(ch, 4*k-1), cw + 5
    max_h = 30 - bank_h - 1
    min_capacity = k + (witness == 1)
    options = [(rh, cl) for rh in range(1, (max_h-3)//(n+1)+1)
               for cl in range(1, 27//(n+1)+1) if rh*cl >= min_capacity]
    _require(bool(options), "unpackable geometry")
    weights = np.array([1 / (rh * cl)**.55 for rh, cl in options])
    rh, cl = options[int(rng.choice(len(options), p=weights/weights.sum()))]
    bh = rh * (n + 1) + 3
    bw = cl * (n + 1) + 3
    bh += int(rng.integers(min(3, max_h-bh)+1))
    bw += int(rng.integers(min(3, 30-bw)+1))
    height, width = bank_h + 1 + bh, max(bank_w, bw)
    extra_h = int(rng.integers(min(2, 30-height)+1))
    extra_w = int(rng.integers(min(2, 30-width)+1))
    top, left = int(rng.integers(extra_h+1)), int(rng.integers(extra_w+1))
    grid = np.full((height+extra_h, width+extra_w), bg, dtype=np.int8)
    bank_left = left + int(rng.integers(width-bank_w+1))
    if rng.random() < .5:
        cat_left, key_left = bank_left, bank_left + cw + 1
    else:
        key_left, cat_left = bank_left, bank_left + 5
    cat_top = top + int(rng.integers(bank_h-ch+1))
    grid[cat_top:cat_top+ch, cat_left:cat_left+cw] = catalogue
    order = list(map(int, rng.permutation(sources)))
    if witness == 1 and order == sources:
        order = order[::-1]
    key_slack = bank_h - (4*k-1)
    gaps = rng.multinomial(key_slack, np.full(k+1, 1/(k+1)))
    key_top = top + int(gaps[0])
    for i, source in enumerate(order):
        key = np.where(_ARROW, source, mapping[source]).astype(np.int8)
        key = np.rot90(key, 2 * int(rng.integers(2)))
        grid[key_top:key_top+3, key_left:key_left+4] = key
        key_top += 4 + int(gaps[i+1])
    board_top = top + bank_h + 1
    board_left = left + int(rng.integers(width-bw+1))
    board = grid[board_top:board_top+bh, board_left:board_left+bw]
    _frame(board, frame_color)
    count = int(rng.integers(min_capacity, min(12, rh*cl)+1))
    row_starts, col_starts = _starts(rng, bh, rh, n), _starts(rng, bw, cl, n)
    slots = [(r, c) for r in row_starts for c in col_starts]
    spots = [slots[int(i)] for i in rng.permutation(len(slots))[:count]]
    for i in rng.permutation(count):
        r, c = spots[int(i)]
        for _ in range(4):
            rr = int(np.clip(r + rng.integers(-2, 3), 2, bh-n-2))
            cc = int(np.clip(c + rng.integers(-2, 3), 2, bw-n-2))
            if all(j == i or rr+n < y or y+n < rr or cc+n < x or x+n < cc
                   for j, (y, x) in enumerate(spots)):
                spots[int(i)] = (rr, cc)
                break
    assignments = sources + list(map(int, rng.choice(sources, count-k)))
    rng.shuffle(assignments)
    output = board.copy()
    for (r, c), source in zip(spots, assignments):
        board[r:r+n, c:c+n] = source
        block = output[r:r+n, c:c+n]
        block[:] = bg
        plus, minus = masks[source]
        block[minus], block[plus] = source, mapping[source]
    return {"input": np.rot90(grid, q).copy(), "output": np.rot90(output, q).copy()}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    _require(nt > ne, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["scale", "copies", "categories", "side"]))
    base_q = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            training = split == "train"
            witness = index + 1 if training and index < 2 else 0
            for _ in range(16):
                if shift == "scale":
                    n = int(rng.choice([2, 3] if training else [4, 5]))
                elif shift == "copies":
                    n = int(rng.choice([2, 3]))
                elif shift == "categories":
                    n = int(rng.choice([2, 3, 4]))
                else:
                    n = int(rng.choice([2, 3, 4, 5]))
                if witness == 1:
                    n = 3
                if shift == "categories":
                    k = int(rng.choice([1, 2] if training else [3, 4]))
                else:
                    k = int(rng.integers(1, (3 if n == 5 else 4)+1))
                if witness:
                    k = 2
                max_copies = min(6, 24//(n+1))
                if shift == "copies":
                    copies = int(rng.choice([3, 4] if training else [5, 6]))
                else:
                    copies = int(rng.integers(3, max_copies+1))
                q = ((base_q if training else (base_q + int(rng.choice([1, 2, 3]))) % 4)
                     if shift == "side" else int(rng.integers(4)))
                pair = _make_pair(rng, n, k, copies, q, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retry bound exhausted")
    # Witness order is not part of the puzzle's rule.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, 2, 10, "training count")
    _count(ne, 1, 4, "test count")
    _require(nt > ne, "training count must exceed test count")
    seen, analyses = set(), {"train": [], "test": []}
    witnesses = {"balanced": False, "mixed": False, "add": False, "remove": False,
                 "asymmetric": False, "columns": False, "repeated": False,
                 "no_comparison": False, "no_mapping": False, "unsigned": False, "recursive": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(a)
            _require(np.array_equal(_render(info), out), "incorrect signed-plate output")
            analyses[split].append(info)
            if split == "train":
                for t in info["entries"].values():
                    p, m = int(t["plus"].sum()), int(t["minus"].sum())
                    witnesses["balanced"] |= p == m and p > 0
                    witnesses["mixed"] |= p > 0 and m > 0
                    witnesses["add"] |= p > 0 and m == 0
                    witnesses["remove"] |= p == 0 and m > 0
                    witnesses["asymmetric"] |= _signed_asymmetric(t["plus"], t["minus"])
                witnesses["columns"] |= len({t["index"] for t in info["entries"].values()}) > 1
                witnesses["repeated"] |= len(info["places"]) > len(info["entries"])
                for name, mode in (("no_comparison", "odd"), ("no_mapping", "no_mapping"),
                                   ("unsigned", "unsigned"), ("recursive", "recursive")):
                    witnesses[name] |= not np.array_equal(_render(info, mode), out)
    _require(all(witnesses.values()), "training evidence missing: " + str([k for k, v in witnesses.items() if not v]))
    tr, te = analyses["train"], analyses["test"]
    shift = (min(t["n"] for t in te) > max(t["n"] for t in tr)
             or min(t["copies"] for t in te) > max(t["copies"] for t in tr)
             or min(len(t["entries"]) for t in te) > max(len(t["entries"]) for t in tr)
             or {t["side"] for t in te}.isdisjoint({t["side"] for t in tr}))
    _require(shift, "missing held-out visible factor")
    return True


def self_test():
    """Bounded streaming checks; no filesystem access and no stored executions."""
    signatures, structures, shapes, scales, counts, copies, sides = set(), set(), set(), set(), set(), set(), set()
    backgrounds, masters, changes = set(), set(), set()
    metrics = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _parse(a)
                scales.add(info["n"])
                counts.add(len(info["entries"]))
                copies.add(info["copies"])
                sides.add(info["side"])
                backgrounds.add(info["bg"])
                shapes.add((a.shape, out.shape))
                roles, normalized = {}, []
                for color in a.flat:
                    normalized.append(roles.setdefault(int(color), len(roles)))
                structures.add(hashlib.sha256(bytes(a.shape) + bytes(normalized)).hexdigest())
                metrics["pairs"] += 1
                metrics["repeated_plates"] += len(info["places"]) > len(info["entries"])
                for t in info["entries"].values():
                    masters.add(_key(t["master"]))
                    changes.add((_key(t["plus"]), _key(t["minus"])))
                    metrics["disconnected_changes"] += len(_components(t["plus"] | t["minus"])) > 1
                    metrics["balanced_changes"] += t["plus"].sum() == t["minus"].sum() > 0
                    metrics["asymmetric_changes"] += _signed_asymmetric(t["plus"], t["minus"])
        # Input-only equivariance and changing-key locality on a representative.
        pair = puzzle["train"][0]
        info = _parse(pair["input"])
        for q in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair["input"], q)), np.rot90(pair["output"], q)),
                     "rotation equivariance failed")
        perm = np.random.default_rng(seed + 42000).permutation(10)
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "palette equivariance failed")
        # Move the exceptional tile to a majority tile's place: print is unchanged.
        t = next(iter(info["entries"].values()))
        odd_i, other_i = t["index"], (t["index"]+1) % t["copies"]
        altered = pair["input"].copy()
        b1, b2 = t["panels"][odd_i][1], t["panels"][other_i][1]
        saved = _slice(altered, b1).copy()
        _slice(altered, b1)[:] = _slice(altered, b2)
        _slice(altered, b2)[:] = saved
        _require(np.array_equal(solve(altered), pair["output"]), "exception relocation failed")
        source = next((s for s, t in info["entries"].items() if t["plus"].any()), None)
        if source is not None:
            target = info["mapping"][source]
            replacement = next(v for v in range(10) if v not in
                               (info["bg"], info["frame"], info["lattice"], source, target))
            altered = pair["input"].copy()
            tile = _slice(altered, info["keys"][source])
            tile[tile == target] = replacement
            expected = pair["output"].copy()
            r0, _, c0, _ = info["frame_box"]
            for r, c, s in info["places"]:
                if s == source:
                    expected[r-r0:r-r0+info["n"], c-c0:c-c0+info["n"]][info["entries"][s]["plus"]] = replacement
            _require(np.array_equal(solve(altered), expected), "key locality failed")
    _require(len(signatures) == 42 and len(structures) >= 200, "insufficient structural variation")
    _require(scales == {2, 3, 4, 5} and counts == {1, 2, 3, 4} and copies == {3, 4, 5, 6},
             "missing geometric variation")
    _require(sides == {"above", "below", "left", "right"} and backgrounds == set(range(10)), "missing role variation")
    _require(len(shapes) >= 80 and len(masters) >= 50 and len(changes) >= 100, "insufficient shape variation")
    _require(metrics["disconnected_changes"] >= 30 and metrics["balanced_changes"] >= 42
             and metrics["asymmetric_changes"] >= 42, "missing edit diversity")
    calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(10000 + 10*nt+ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "wrong explicit counts")
            calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    base = generate_puzzle(0, 2, 1)
    for defect in ("label", "absent_exception", "arrow"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = bad["train"][0]
        info = _parse(pair["input"])
        if defect == "label":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif defect == "absent_exception":
            source, t = next(iter(info["entries"].items()))
            _slice(pair["input"], t["panels"][t["index"]][1])[:] = np.where(t["master"], source, info["bg"])
        else:
            box = next(iter(info["keys"].values()))
            pair["input"][box[0], box[2]] = info["bg"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("invalid mutation accepted: " + defect)
    return dict(valid=True, seeds=42, pairs=metrics["pairs"], explicit_count_combinations=calls,
                distinct_puzzles=len(signatures), normalized_inputs=len(structures), shape_pairs=len(shapes),
                master_bitmaps=len(masters), signed_stencils=len(changes),
                edit_metrics={k: int(v) for k, v in metrics.items()},
                panel_sides=sorted(scales), categories=sorted(counts), copies=sorted(copies), sides=sorted(sides),
                independent_solver=False, blind_solve=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
