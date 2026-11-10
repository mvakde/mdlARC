"""Powered jigsaw: rectangular closure is constrained by branching-ray reachability.

Only NumPy and the standard library are used. No parent code or data are imported.
All search limits reject incomplete searches; they never decide an assembly.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


class _Reject(ValueError):
    """A candidate is ambiguous, unsupported, or exceeds the finite search budget."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells = [first]
        for y, x in cells:
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
        groups.append(cells)
    return groups


def _marker(point, corner):
    y, x = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((y+dy, x), (y, x+dx), (y+dy, x+dx))


def _prefix(board, guide):
    """Check a downward, two-cell-deep, one/two-source boundary prefix."""
    if board.shape[0] < 3:
        return False
    cols = np.flatnonzero(board[0] == guide)
    return (len(cols) in (1, 2) and bool(np.all(np.diff(cols) >= 2))
            and np.array_equal(cols, np.flatnonzero(board[1] == guide))
            and not bool(np.any(board[2:] == guide)))


def _extract(a):
    """Roles are visible: 0 outside, 5 barriers, external L, short source, carrier."""
    counts = np.bincount(a.ravel(), minlength=10)
    markers = []
    for color in range(1, 10):
        if color == 5 or counts[color] != 3:
            continue
        q = np.argwhere(a == color)
        low, high = q.min(0), q.max(0)
        if tuple(high-low) != (1, 1):
            continue
        square = {(int(y), int(x)) for y in (low[0], high[0]) for x in (low[1], high[1])}
        missing = square - set(map(tuple, q.tolist()))
        point = missing.pop()
        if a[point] in (0, color):
            continue
        dy, dx = point[0]-int(low[0]), point[1]-int(low[1])
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[(dy, dx)]
        markers.append((color, point, corner))
    _require(len(markers) == 1, "one external L marker required")
    marker, point, corner = markers[0]
    payload = [c for c in range(1, 10) if c not in (marker, 5) and counts[c]]
    _require(len(payload) == 3 and counts[5] >= 6, "payload roles")
    guides = [c for c in payload if counts[c] in (2, 4)]
    _require(len(guides) == 1, "source must contain two or four cells")
    guide = guides[0]
    others = [c for c in payload if c != guide]
    carrier = max(others, key=lambda c: counts[c])
    receiver = next(c for c in others if c != carrier)
    _require(counts[receiver] >= 6 and counts[carrier] > max(counts[c] for c in range(1, 10) if c != carrier),
             "unique carrier and at least six receivers required")
    pieces = []
    anchor = None
    for cells in _components((a != 0) & (a != marker)):
        q = np.array(cells)
        y, x = map(int, q.min(0))
        end_y, end_x = map(int, q.max(0)+1)
        matrix = np.full((end_y-y, end_x-x), -1, dtype=np.int8)
        for r, c in cells:
            matrix[r-y, c-x] = a[r, c]
        if point in cells:
            anchor = len(pieces)
        pieces.append((matrix, (y, x)))
    _require(anchor is not None and 3 <= len(pieces) <= 6, "three to six payload pieces")
    matrix, (ay, ax) = pieces[anchor]
    expected = (ay + (matrix.shape[0]-1 if corner[0] == "B" else 0),
                ax + (matrix.shape[1]-1 if corner[1] == "R" else 0))
    _require(point == expected, "marker must pin a bounding-box corner")
    _require(all(i == anchor or not np.any(m == guide) for i, (m, _) in enumerate(pieces)),
             "all sources belong to anchor")
    turns = [k for k in range(4) if _prefix(np.rot90(matrix, k), guide)]
    _require(len(turns) == 1, "source direction must be unique")
    return dict(marker=marker, point=point, corner=corner, carrier=carrier,
                guide=guide, receiver=receiver, pieces=pieces, anchor=anchor, turns=turns[0])


def _packings(info, canvas):
    """Complete translation-only mask tilings, over every feasible rectangle shape.

    At the first uncovered cell the next piece's first occupied cell must land.
    Bit masks test occupied cells, including complementary steps. Scattered
    positions of non-anchor pieces never participate in this search.
    """
    pieces, anchor = info["pieces"], info["anchor"]
    area = sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces)
    answer = []
    nodes = 0
    py, px = info["point"]
    corner = info["corner"]
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        oy = py - (h-1 if corner[0] == "B" else 0)
        ox = px - (w-1 if corner[1] == "R" else 0)
        if w > canvas[1] or min(oy, ox) < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        configs = []
        for m, _ in pieces:
            cells = [tuple(map(int, p)) for p in np.argwhere(m >= 0)]
            bits = sum(1 << (y*w+x) for y, x in cells)
            configs.append((m.shape, cells[0][1], bits))
        ay, ax = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        ah, aw = pieces[anchor][0].shape
        if min(ay, ax) < 0 or ay+ah > h or ax+aw > w:
            continue
        positions = [None]*len(pieces)
        positions[anchor] = (ay, ax)
        full = (1 << area)-1

        def visit(occupied, todo):
            nonlocal nodes
            nodes += 1
            if nodes > 24000 or len(answer) >= 1200:
                raise _Reject("complete packing search exceeded budget")
            if not todo:
                answer.append(((h, w), (oy, ox), tuple(positions)))
                return
            free = full ^ occupied
            y, x = divmod((free & -free).bit_length()-1, w)
            for i in todo:
                (ph, pw), first_x, bits = configs[i]
                left = x-first_x
                if left < 0 or y+ph > h or left+pw > w:
                    continue
                placed = bits << (y*w+left)
                if not placed & occupied:
                    positions[i] = (y, left)
                    visit(occupied | placed, tuple(j for j in todo if j != i))

        visit(configs[anchor][2] << (ay*w+ax), tuple(i for i in range(len(pieces)) if i != anchor))
    return answer


def _assemble(info, candidate):
    shape, _, positions = candidate
    board = np.full(shape, -1, dtype=np.int8)
    labels = np.full(shape, -1, dtype=np.int8)
    identifiers = np.full(shape, -1, dtype=np.int16)
    offset = 0
    for i, ((m, _), (y, x)) in enumerate(zip(info["pieces"], positions)):
        mask = m >= 0
        sub = board[y:y+m.shape[0], x:x+m.shape[1]]
        sub[mask] = m[mask]
        labels[y:y+m.shape[0], x:x+m.shape[1]][mask] = i
        n = int(mask.sum())
        identifiers[y:y+m.shape[0], x:x+m.shape[1]][mask] = np.arange(offset, offset+n)
        offset += n
    assert np.all(board >= 0)
    return board, labels, identifiers


def _route(board, guide, turns=0, labels=None, shortcut=None):
    a = np.rot90(board, turns)
    if not _prefix(a, guide):
        raise _Reject("assembled source is not on the incoming boundary")
    lab = np.rot90(labels, turns) if labels is not None else np.zeros_like(a)
    out = a.copy()
    trace = a == guide
    active = set(map(int, np.flatnonzero(a[0] == guide)))
    h, w = a.shape
    events = []
    all_runs = set()
    for r in range(h):
        c = 0
        while c < w:
            if a[r, c] == 5:
                left = c
                while c+1 < w and a[r, c+1] == 5:
                    c += 1
                all_runs.add((r, left, c))
            c += 1
    for r in range(1, h):
        hits = {}
        following = set()
        for c in sorted(active):
            if a[r, c] != 5:
                trace[r, c] = True
                following.add(c)
                continue
            if shortcut == "straight":
                continue
            left = right = c
            while left and a[r, left-1] == 5 and (shortcut != "piece_runs" or lab[r, left-1] == lab[r, c]):
                left -= 1
            while right+1 < w and a[r, right+1] == 5 and (shortcut != "piece_runs" or lab[r, right+1] == lab[r, c]):
                right += 1
            if shortcut == "ignore_singletons" and left == right:
                continue
            hits[(left, right)] = True
        if shortcut == "all_bars":
            hits = {(left, right): True for rr, left, right in all_runs if rr == r}
        if shortcut == "one_per_layer":
            hits = dict(list(sorted(hits.items()))[:1])
        for left, right in hits:
            events.append((r, left, right))
            for c in range(max(0, left-1), min(w, right+2)):
                if a[r-1, c] != 5:
                    trace[r-1, c] = True
            for c in (left-1, right+1):
                if 0 <= c < w and a[r, c] != 5:
                    trace[r, c] = True
                    following.add(c)
        active = following
    out[trace] = guide
    stats = dict(events=events, reached=len(set(events)), inert=len(all_runs-set(events)),
                 joined=sum(len(set(map(int, lab[r, left:right+1]))) > 1 for r, left, right in events),
                 simultaneous=sum(n > 1 for n in Counter(r for r, _, _ in events).values()),
                 clipped=sum(left == 0 or right == w-1 for _, left, right in events),
                 two_flank=sum(left > 0 and right < w-1 for _, left, right in events),
                 sources=len(np.flatnonzero(a[0] == guide)))
    return np.rot90(out, -turns).copy(), np.rot90(trace, -turns).copy(), stats


def _infer(a):
    info = _extract(a)
    candidates = _packings(info, a.shape)
    winner = None
    outputs = set()
    feasible = 0
    for candidate in candidates:
        board, labels, _ = _assemble(info, candidate)
        try:
            lit, trace, stats = _route(board, info["guide"], info["turns"], labels)
        except _Reject:
            continue
        if not np.all(trace[board == info["receiver"]]):
            continue
        out = np.zeros_like(a)
        (h, w), (y, x), _ = candidate
        out[y:y+h, x:x+w] = lit
        outputs.add(_key(out))
        feasible += 1
        if len(outputs) > 1:
            raise _Reject("different powered assemblies remain")
        winner = (out, candidate, board, labels, trace, stats)
    if winner is None:
        raise _Reject("no assembly supplies every receiver")
    return info, candidates, feasible, winner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this grid alone; no caches or latent state."""
    return _infer(_grid(input_grid))[3][0].copy()


def _parts(rng, total, count, first_min=2):
    parts = [first_min] + [2]*(count-1)
    if sum(parts) > total:
        raise _Reject("partition too small")
    for _ in range(total-sum(parts)):
        parts[int(rng.integers(count))] += 1
    return parts


def _partition(rng, h, w, n, stepped):
    bands = 3 if w >= 9 and n >= 4 and rng.random() < .4 else 2
    counts = [1]*bands
    for _ in range(n-bands):
        choices = [i for i in range(bands) if counts[i] < (h-1)//2]
        counts[int(rng.choice(choices))] += 1
    widths = _parts(rng, w, bands, 3)
    labels = np.full((h, w), -1, dtype=np.int8)
    i, x = 0, 0
    for b, (bw, count) in enumerate(zip(widths, counts)):
        y = 0
        for bh in _parts(rng, h, count, 3 if b == 0 else 2):
            labels[y:y+bh, x:x+bw] = i
            y += bh
            i += 1
        x += bw
    if stepped:
        options = []
        for y in range(2, h-1):
            for x in range(1, w-1):
                for dy, dx in ((0, 1), (1, 0)):
                    if labels[y, x] != labels[y+dy, x+dx]:
                        options.append((y, x, int(labels[y+dy, x+dx])))
        rng.shuffle(options)
        changed = False
        for y, x, recipient in options[:40]:
            donor = int(labels[y, x])
            labels[y, x] = recipient
            if all(len(_components(labels == i)) == 1 for i in (donor, recipient)):
                # A transfer is useful only when it makes a genuine nonrectangular mask.
                if any(len(c := np.argwhere(labels == i)) < int(np.prod(np.ptp(c, axis=0)+1)) for i in (donor, recipient)):
                    changed = True
                    break
            labels[y, x] = donor
        if not changed:
            raise _Reject("no connected step")
    return labels


def _pieces(board, labels, origin):
    result = []
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i)
        y, x = map(int, q.min(0))
        ey, ex = map(int, q.max(0)+1)
        m = np.where(labels[y:ey, x:ex] == i, board[y:ey, x:ex], -1).astype(np.int8)
        result.append((m, (origin[0]+y, origin[1]+x)))
    return result


def _scatter(rng, pieces, anchor, canvas, markers):
    """Bounded placement with a blank row or column between bounding boxes."""
    order = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i][0].size)
    for _ in range(10):
        positions = {anchor: pieces[anchor][1]}
        placed = [(*pieces[anchor][1], *pieces[anchor][0].shape)]
        for i in order:
            ph, pw = pieces[i][0].shape
            options = [(y, x) for y in range(canvas[0]-ph+1) for x in range(canvas[1]-pw+1)]
            rng.shuffle(options)
            for y, x in options:
                if any(y-1 <= r <= y+ph and x-1 <= c <= x+pw for r, c in markers):
                    continue
                if all(y+ph < r or r+rh < y or x+pw < c or c+cw < x for r, c, rh, cw in placed):
                    positions[i] = (y, x)
                    placed.append((y, x, ph, pw))
                    break
            else:
                break
        if len(positions) == len(pieces):
            return [positions[i] for i in range(len(pieces))]
    raise _Reject("scatter capacity exhausted")


def _choose_receivers(rng, info, candidates, intended, labels):
    """Positive reachability clues eliminate wrong answers, not hidden piece codes."""
    board, _, ids = _assemble(info, intended)
    correct, trace, stats = _route(board, info["guide"], labels=labels)
    if stats["reached"] < 2 or not stats["joined"] or not stats["inert"]:
        raise _Reject("need a causal cascade, a rejoined barrier and an inert obstacle")
    piece_wrong = _route(board, info["guide"], labels=labels, shortcut="piece_runs")[0]
    if np.array_equal(correct, piece_wrong):
        raise _Reject("cut-spanning collision is visually redundant")
    eligible = set(map(int, ids[trace & (board != info["guide"])]))
    if len(eligible) < 12:
        raise _Reject("too little unhinted path")
    differences = []
    for candidate in candidates:
        alt, alabels, aids = _assemble(info, candidate)
        lit, atrace, _ = _route(alt, info["guide"], labels=alabels)
        # Include rectangle placement in the answer identity.
        sig = (candidate[:2], lit.tobytes())
        intended_sig = (intended[:2], correct.tobytes())
        if sig == intended_sig:
            continue
        missing = eligible - set(map(int, aids[atrace]))
        if not missing:
            raise _Reject("positive receivers cannot disambiguate this board")
        differences.append(missing)
    if not differences:
        raise _Reject("geometry alone determines the answer")
    chosen = set()
    while differences:
        scores = Counter(i for d in differences for i in d)
        best = max(scores.values())
        options = sorted(i for i, n in scores.items() if n == best)
        pick = int(rng.choice(options))
        chosen.add(pick)
        differences = [d for d in differences if pick not in d]
        if len(chosen) > 18:
            raise _Reject("too many receiver clues needed")
    for i in range(len(info["pieces"])):
        options = sorted(eligible & set(map(int, ids[labels == i])))
        if not options:
            raise _Reject("each piece must receive power")
        if not chosen.intersection(options):
            chosen.add(int(rng.choice(options)))
    target = min(18, max(6, len(chosen)+int(rng.integers(1, 4))))
    extra = sorted(eligible-chosen)
    if len(chosen) > 18 or len(eligible)-target < 5:
        raise _Reject("clues would reveal the entire trace")
    if target > len(chosen):
        chosen.update(map(int, rng.choice(extra, target-len(chosen), replace=False)))
    board[np.isin(ids, sorted(chosen))] = info["receiver"]
    return board, correct


def _make_pair(rng, n, stepped, testing, palette, rotation, reflected, allowed=None, witness=False):
    for _ in range(100):
        try:
            h = int(rng.integers(14, 18) if testing else rng.integers(9, 14))
            w = int(rng.integers(8, 13) if testing else rng.integers(7, 12))
            labels = _partition(rng, h, w, n, stepped)
            board = np.ones((h, w), dtype=np.int8)
            anchor_width = int(np.count_nonzero(labels[0] == 0))
            sources = [int(rng.integers(1 if witness else 0, anchor_width))]
            if not witness and anchor_width >= 4 and rng.random() < .4:
                choices = [c for c in range(anchor_width) if abs(c-sources[0]) >= 2]
                sources.append(int(rng.choice(choices)))
            if any(np.any(labels[:2, c] != 0) for c in sources):
                continue
            board[:2, sources] = 3
            first = int(rng.integers(2, min(6, h-3)))
            left = int(rng.integers(1 if witness else 0, min(sources)+1))
            right = int(rng.integers(max(max(sources), anchor_width), w-1 if witness else w))
            board[first, left:right+1] = 5
            for _ in range(int(rng.integers(4, 9))):
                r = int(rng.integers(2, h))
                length = int(rng.integers(1, w))
                c = int(rng.integers(w-length+1))
                board[r, c:c+length] = 5
                if rng.random() < .25 and r+1 < h:
                    board[r+1:min(h, r+3), c] = 5
            # This same-color prefix bar is unreachable and makes causal selection visible.
            inert_cols = [c for c in range(w) if c not in sources and all(abs(c-s) > 1 for s in sources)]
            if inert_cols:
                board[0, int(rng.choice(inert_cols))] = 5
            if np.count_nonzero(board == 5) < 6:
                continue
            area = h*w
            ch = min(30, max(h+3, int(np.ceil(np.sqrt(area*2.7))))+int(rng.integers(0, 4)))
            cw = min(30, max(w+3, int(np.ceil(area*2.7/ch)))+int(rng.integers(0, 4)))
            canvas = (ch, cw)
            origin = (int(rng.integers(1, ch-h)), int(rng.integers(1, cw-w)))
            pieces = _pieces(board, labels, origin)
            info = dict(pieces=pieces, anchor=0, point=origin, corner="TL", guide=3, receiver=4)
            candidates = _packings(info, canvas)
            intended = ((h, w), origin, tuple((y-origin[0], x-origin[1]) for _, (y, x) in pieces))
            if intended not in candidates:
                raise _Reject("partition lost its anchor")
            hinted, expected = _choose_receivers(rng, info, candidates, intended, labels)
            counts = np.bincount(hinted.ravel(), minlength=6)
            if counts[1] <= max(counts[3], counts[4], counts[5]):
                continue
            pieces = _pieces(hinted, labels, origin)
            marks = _marker(origin, "TL")
            positions = _scatter(rng, pieces, 0, canvas, marks)
            a = np.zeros(canvas, dtype=np.int8)
            for (m, _), (y, x) in zip(pieces, positions):
                a[y:y+m.shape[0], x:x+m.shape[1]][m >= 0] = m[m >= 0]
            for p in marks:
                a[p] = 2
            out = np.zeros(canvas, dtype=np.int8)
            out[origin[0]:origin[0]+h, origin[1]:origin[1]+w] = expected
            mapping = np.array([0, palette[0], palette[1], palette[2], palette[3], 5], dtype=np.int8)
            a, out = mapping[a], mapping[out]
            if reflected:
                a, out = np.fliplr(a), np.fliplr(out)
            a, out = np.rot90(a, rotation).copy(), np.rot90(out, rotation).copy()
            pair = {"input": a, "output": out}
            try:
                measured = _pair_info(pair)
            except ValueError as exc:
                raise _Reject(str(exc)) from exc
            if witness and not measured["two_flank"]:
                continue
            if allowed is not None and any(measured["wrong"][name] and name not in allowed for name in ("one_per_layer", "ignore_singletons")):
                continue
            return pair
        except _Reject:
            continue
    raise RuntimeError("bounded candidate construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= value <= high, name+" out of range or not an integer")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    palette = tuple(map(int, rng.choice([1, 2, 3, 4, 6, 7, 8, 9], 4, replace=False)))
    mixed = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            testing = split == "test"
            n = int(rng.integers(5, 7) if testing else 3 if i == 0 else 4 if i == 1 else rng.integers(3, 6))
            stepped = i == 1 if not testing else bool(rng.integers(2))
            if not testing and i > 1:
                stepped = n >= 4 and bool(rng.integers(2))
            local_palette = palette
            if mixed:
                local_palette = tuple(map(int, rng.choice([1, 2, 3, 4, 6, 7, 8, 9], 4, replace=False)))
            for _ in range(8):
                pair = _make_pair(rng, n, stepped, testing, local_palette,
                                  (base_rotation+i+int(testing)) % 4, bool(rng.integers(2)),
                                  allowed=({name for name in ("one_per_layer", "ignore_singletons")
                                            if any(_pair_info(p)["wrong"][name] for p in puzzle["train"])} if testing else None),
                                  witness=not testing and i == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "canvas must be retained")
    info, candidates, feasible, winner = _infer(a)
    predicted, candidate, board, labels, trace, stats = winner
    _require(np.array_equal(predicted, out), "incorrect output")
    _require(not np.any(out == info["marker"]), "erase marker")
    _require(all(out[p] == 0 for p in _marker(info["point"], info["corner"])), "marker sites must be empty")
    _require(not np.any(out == info["receiver"]), "all receivers must be supplied")
    _require(stats["reached"] >= 2 and stats["inert"] > 0 and stats["joined"] > 0,
             "missing causal or cross-piece event")
    shape, origin, positions = candidate
    (ay, ax) = positions[info["anchor"]]
    _require((origin[0]+ay, origin[1]+ax) == info["pieces"][info["anchor"]][1], "anchor moved")
    _require(all(np.any(trace & (labels == i)) for i in range(len(info["pieces"]))), "unpowered piece")
    _require(all(np.any(m == info["receiver"]) for m, _ in info["pieces"]), "each piece needs a receiver")
    _require(np.count_nonzero(trace & (board != info["guide"]) & (board != info["receiver"])) >= 5,
             "receiver dots must not reveal the whole trace")
    _require(np.count_nonzero(a) == np.count_nonzero(out)+3, "payload area must be conserved")
    _require(np.count_nonzero(out == 0) == np.count_nonzero(a == 0)+3, "vacated sites and marker restoration")
    for color in (5,):
        _require(np.count_nonzero(a == color) == np.count_nonzero(out == color), "barriers must be conserved")
    _require(np.all(out[origin[0]:origin[0]+shape[0], origin[1]:origin[1]+shape[1]][board == 5] == 5), "barrier recolored")
    wrong = {}
    for name in ("straight", "piece_runs", "all_bars", "one_per_layer", "ignore_singletons"):
        alt = _route(board, info["guide"], info["turns"], labels, name)[0]
        wrong[name] = not np.array_equal(alt, predicted[origin[0]:origin[0]+shape[0], origin[1]:origin[1]+shape[1]])
    _require(wrong["straight"] and wrong["piece_runs"] and wrong["all_bars"], "parent-A shortcut not distinguished")
    _require(len(candidates) >= 2 and feasible < len(candidates), "ray evidence must constrain assembly")
    geometric_outputs = set()
    for c in candidates:
        b, labs, _ = _assemble(info, c)
        b[b == info["receiver"]] = info["carrier"]
        lit = _route(b, info["guide"], info["turns"], labs)[0]
        geometric_outputs.add((c[:2], lit.tobytes()))
    _require(len(geometric_outputs) >= 2, "geometry-only ambiguity must change the answer")
    # Removing B entirely leaves the scattered canvas, which is never the answer.
    _require(not np.array_equal(a, out), "assembly must change the answer")
    stepped = sum(int(np.count_nonzero(m >= 0)) < m.size for m, _ in info["pieces"])
    stats.update(shape=shape, canvas=a.shape, corner=info["corner"], count=len(info["pieces"]),
                 steps=stepped, geometric=len(candidates), feasible=feasible,
                 depth=np.rot90(board, info["turns"]).shape[0],
                 wrong=wrong, palette=tuple(info[k] for k in ("carrier", "marker", "guide", "receiver")))
    return stats


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            stats[split].append(_pair_info(pair))
    _require(any(s["steps"] for s in stats["train"]), "steps need a training demonstration")
    _require(any(s["two_flank"] for s in stats["train"]), "training must visibly branch both ways")
    _require(len({s["count"] for s in stats["train"]}) >= 2, "training must vary partition size")
    _require(min(s["depth"] for s in stats["test"]) > max(s["depth"] for s in stats["train"]), "depth shift")
    for shortcut in ("one_per_layer", "ignore_singletons"):
        _require(any(s["wrong"][shortcut] for s in stats["train"])
                 or not any(s["wrong"][shortcut] for s in stats["test"]), "undemonstrated local collision case")
    return None


def self_test():
    """Stream 42 seeds; keep only hashes, counters and compact feature sets."""
    # Hand-specified local collision regression, independent of random construction.
    board = np.ones((7, 7), dtype=np.int8)
    board[:2, 3] = 3
    board[3, 2:5] = 5
    board[5, [1, 5]] = 4
    expected_rows = ("1113111", "1113111", "1333331", "1355531", "1311131", "1311131", "1311131")
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=np.int8)
    _require(np.array_equal(_route(board, 3)[0], expected), "local collision regression")
    signatures, shapes, partitions, palettes, corners, counts = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                s = _pair_info(pair)
                shapes.add(s["canvas"])
                palettes.add(s["palette"])
                corners.add(s["corner"])
                counts.add(s["count"])
                inf, _, _, win = _infer(a)
                lab = np.rot90(win[3], inf["turns"])
                rename = {}
                normalized = bytes(rename.setdefault(int(c), len(rename)) for c in lab.ravel())
                partitions.add((lab.shape, normalized))
                for name in ("joined", "simultaneous", "clipped", "steps"):
                    totals[name] += int(bool(s[name]))
                totals[split+"_pairs"] += 1
                # Geometric transformations must commute with the input-only solver.
                if seed < 4 and split == "test":
                    for k in range(4):
                        _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(pair["output"], k)), "rotation equivariance")
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(partitions) >= 60, "structural diversity")
    _require(len(palettes) >= 20 and corners == {"TL", "TR", "BL", "BR"} and counts == {3, 4, 5, 6}, "role/count diversity")
    p = generate_puzzle(17, 2, 1)
    q = generate_puzzle(17, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "determinism")
    # Fresh scattering changes extraction order and locations while retaining the answer.
    pair = p["test"][0]
    a = pair["input"]
    inf = _extract(a)
    marks = _marker(inf["point"], inf["corner"])
    positions = _scatter(np.random.default_rng(610), inf["pieces"], inf["anchor"], a.shape, marks)
    shuffled = np.zeros_like(a)
    for (m, _), (y, x) in zip(inf["pieces"], positions):
        shuffled[y:y+m.shape[0], x:x+m.shape[1]][m >= 0] = m[m >= 0]
    for point in marks:
        shuffled[point] = inf["marker"]
    _require(np.array_equal(solve(shuffled), pair["output"]), "scattering must not encode order")
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for corruption in ("marker", "barrier", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        a = damaged["test"][0]["input"]
        out = damaged["test"][0]["output"]
        if corruption == "marker":
            inf = _extract(a)
            out[_marker(inf["point"], inf["corner"])[0]] = inf["marker"]
        elif corruption == "barrier":
            out[tuple(np.argwhere(out == 5)[0])] = 0
        else:
            damaged["train"][1] = damaged["train"][0]
        try:
            validate_generated_puzzle(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("corruption accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), canvas_shapes=len(shapes),
                partition_descriptors=len(partitions), palettes=len(palettes),
                corners=sorted(corners), piece_counts=sorted(counts), features=dict(totals))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
