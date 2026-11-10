"""Repaired stencil watersheds. Standard library and NumPy only.

No parent imports, file access, latent puzzle state, or output-based parsing.
All arrays retained by generation are bounded by fourteen 30x30 pairs.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import numpy as np

BG, FRAME, WIRE = 0, 5, 6
LABELS = (1, 2, 3, 4, 7, 8, 9)
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10000


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer color grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _counts(a, b):
    for name, value, low, high in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    return int(a), int(b)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    """Support in at least one intact horizontal/vertical triple."""
    result = np.zeros_like(mask)
    for m, dest in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            dest[:, offset:offset + triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for m, vertical in ((mask, False), (mask.T, True)):
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            result.append((int(x), int(y), 1, 0) if vertical
                          else (int(y), int(x), 0, 1))
    return result


def _recover(mask):
    """Return clean geometry, repair evidence, and erased diagonal cells.

    All evidence is read simultaneously from the unmodified input. An evidence
    record is (bracket origin/direction, two off-axis cells).
    """
    strong = _strong(mask)
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                cells = tuple((ar+i*sr, ac+i*sc) for i in (1, 2))
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in cells)
                        and not _at(mask, ar+3*sr, ac+3*sc)):
                    matches.append(cells)
        if len(matches) > 1:
            raise ValueError("ambiguous diagonal attachment")
        if matches:
            repairs.append(((r, c, dr, dc), matches[0]))
    used_spurs = set()
    for _, cells in repairs:
        if used_spurs.intersection(cells):
            raise ValueError("one diagonal cannot identify two repairs")
        used_spurs.update(cells)
    added, anchors = np.zeros_like(mask), np.zeros_like(mask)
    for (r, c, dr, dc), _ in repairs:
        for i in (1, 2, 3):
            added[r+i*dr, c+i*dc] = True
        anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
    weak = mask & ~_strong(mask | added) & ~anchors
    erased = np.zeros_like(mask)
    for a, b, da, db in (
        (weak[:-1, :-1], weak[1:, 1:], erased[:-1, :-1], erased[1:, 1:]),
        (weak[:-1, 1:], weak[1:, :-1], erased[:-1, 1:], erased[1:, :-1]),
    ):
        linked = a & b
        da |= linked
        db |= linked
    return (mask & ~erased) | added, repairs, erased


def _territory(mask, roots, metric="wire"):
    """One BFS per identity; same-identity terminals cooperate."""
    ids = sorted(roots)
    distances = np.full((len(ids), *mask.shape), INF, dtype=np.int16)
    rr, cc = np.indices(mask.shape)
    for i, label in enumerate(ids):
        d = distances[i]
        if metric == "manhattan":
            for r, c in roots[label]:
                d[:] = np.minimum(d, np.abs(rr-r) + np.abs(cc-c))
            continue
        queue = deque()
        for r, c in roots[label]:
            if not mask[r, c]:
                raise ValueError("terminal outside the recovered wire")
            d[r, c] = 0
            queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in STEPS:
                nr, nc = r+dr, c+dc
                if _at(mask, nr, nc) and d[nr, nc] == INF:
                    d[nr, nc] = d[r, c] + 1
                    queue.append((nr, nc))
    best = distances.min(axis=0)
    unique = (distances == best).sum(axis=0) == 1
    assigned = mask & (best < INF) & unique
    winner = np.full(mask.shape, -1, dtype=np.int8)
    choice = distances.argmin(axis=0)
    for i, label in enumerate(ids):
        winner[assigned & (choice == i)] = label
    ties = mask & (best < INF) & ~unique
    unreachable = mask & (best == INF)
    return winner, ties, unreachable, distances


def _parse(input_grid):
    a = _grid(input_grid)
    if a[0, 0] != FRAME:
        raise ValueError("missing library frame")
    lw = 0
    while lw < a.shape[1] and a[0, lw] == FRAME:
        lw += 1
    divider = next((r for r in range(1, min(5, a.shape[0]))
                    if a[r, 1] == FRAME), None)
    if divider is None:
        raise ValueError("missing motif divider")
    s = divider - 1
    if s not in (2, 3) or (lw-1) % (s+1):
        raise ValueError("invalid library scale")
    k = (lw-1)//(s+1)
    if not 2 <= k <= 3:
        raise ValueError("invalid library count")
    board_top = s+5
    if a.shape[0] < board_top+3:
        raise ValueError("missing map")
    bw = 0
    while bw < a.shape[1] and a[board_top, bw] == FRAME:
        bw += 1
    h, w = a.shape[0]-board_top-2, bw-2
    if not (6 <= h <= 10 and 6 <= w <= 10 and max(h*s, w*s) <= 30):
        raise ValueError("map bounds")
    expected = np.zeros_like(a)
    expected[0, :lw] = expected[s+1, :lw] = expected[s+3, :lw] = FRAME
    for j in range(k+1):
        expected[:s+4, j*(s+1)] = FRAME
    templates, targets, order = {}, {}, []
    for j in range(k):
        c = 1+j*(s+1)
        panel = a[1:s+1, c:c+s]
        values = np.unique(panel[panel != BG])
        if len(values) != 1 or int(values[0]) not in LABELS:
            raise ValueError("a motif must have one visible source identity")
        label = int(values[0])
        if label in templates:
            raise ValueError("duplicate library identity")
        target = int(a[s+2, c+s//2])
        if target == BG:
            raise ValueError("invisible key")
        templates[label] = panel == label
        targets[label] = target
        order.append(label)
        expected[1:s+1, c:c+s] = panel
        expected[s+2, c+s//2] = target
    expected[board_top, :bw] = expected[-1, :bw] = FRAME
    expected[board_top:, 0] = expected[board_top:, bw-1] = FRAME
    board = a[board_top+1:-1, 1:bw-1]
    if not set(map(int, np.unique(board))) <= {BG, WIRE, *order}:
        raise ValueError("unknown wire-map color")
    expected[board_top+1:-1, 1:bw-1] = board
    if not np.array_equal(a, expected):
        raise ValueError("malformed frame, key, spacer, or padding")
    roots = {label: [tuple(map(int, p)) for p in np.argwhere(board == label)]
             for label in order}
    if any(not positions for positions in roots.values()):
        raise ValueError("library identity without a terminal")
    clean, repairs, erased = _recover(board != BG)
    if any(erased[r, c] for positions in roots.values() for r, c in positions):
        raise ValueError("terminal on diagonal clutter")
    if any(board[r, c] != WIRE for _, cells in repairs for r, c in cells):
        raise ValueError("deviation must use the wire color")
    return dict(s=s, k=k, board=board.copy(), clean=clean, repairs=repairs,
                erased=erased, templates=templates, targets=targets,
                order=order, roots=roots)


def _render(parsed, winner, variant="normal"):
    s = parsed["s"]
    output = np.zeros((winner.shape[0]*s, winner.shape[1]*s), dtype=np.int8)
    root_order = list(dict.fromkeys(int(x) for x in parsed["board"].flat
                                   if int(x) in parsed["roots"]))
    for r, c in np.argwhere(winner >= 0):
        label = int(winner[r, c])
        entry = (parsed["order"][root_order.index(label)]
                 if variant == "position" else label)
        mask = parsed["templates"][entry]
        if variant == "solid":
            mask = np.ones_like(mask)
        color = entry if variant == "unkeyed" else parsed["targets"][entry]
        output[r*s:(r+1)*s, c*s:(c+1)*s] = np.where(mask, color, BG)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(input_grid)
    winner, _, _, _ = _territory(parsed["clean"], parsed["roots"])
    return _render(parsed, winner)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in STEPS:
                q = (r+dr, c+dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _thick(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1]
                 & mask[:-1, 1:] & mask[1:, 1:]).any())


def _network(rng, h, w):
    """Random spanning subgraphs on irregularly spaced orthogonal junctions."""
    axes = []
    for length in (h, w):
        ends = [(lo, hi) for lo in range(2) for hi in range(length-2, length)
                if hi-lo >= 6]
        lo, hi = ends[int(rng.integers(len(ends)))]
        coords = [lo, hi]
        if hi-lo >= 6 and rng.random() < .6:
            coords.insert(1, int(rng.integers(lo+3, hi-2)))
        axes.append(coords)
    rows, cols = axes
    nodes = [(r, c) for r in range(len(rows)) for c in range(len(cols))]
    edges = [(p, (p[0]+dr, p[1]+dc)) for p in nodes for dr, dc in ((1, 0), (0, 1))
             if p[0]+dr < len(rows) and p[1]+dc < len(cols)]
    parent = {p: p for p in nodes}

    def find(p):
        while parent[p] != p:
            p = parent[p]
        return p

    chosen, extra = [], []
    for index in rng.permutation(len(edges)):
        p, q = edges[int(index)]
        a, b = find(p), find(q)
        if a != b:
            parent[a] = b
            chosen.append((p, q))
        else:
            extra.append((p, q))
    probability = float(rng.choice([0, .25, .65, 1]))
    chosen += [edge for edge in extra if rng.random() < probability]
    mask = np.zeros((h, w), bool)
    for p, q in chosen:
        r0, c0 = rows[p[0]], cols[p[1]]
        r1, c1 = rows[q[0]], cols[q[1]]
        mask[r0:r1+1, c0:c1+1] = True
    return mask


def _options(mask, with_spur=True):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            if not all(_at(mask, r+i*dr, c+i*dc) for i in range(5)):
                continue
            if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+i*dr, c+i*dc) for i in (1, 2, 3))
            if any(_at(mask, rr+dc, cc+dr) or _at(mask, rr-dc, cc-dr) for rr, cc in gap):
                continue
            if not with_spur:
                result.append((gap, ()))
                continue
            for end in (0, 4):
                ar, ac = r+end*dr, c+end*dc
                toward = 1 if end == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr+side*dc, toward*dc+side*dr
                    spur = tuple((ar+i*sr, ac+i*sc) for i in (1, 2))
                    if all(0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                           and not mask[rr, cc] for rr, cc in spur):
                        result.append((gap, spur))
    return result


def _corrupt(rng, clean, count):
    inp = clean.copy()
    options = _options(clean)
    done = 0
    for index in rng.permutation(len(options)):
        gap, spur = options[int(index)]
        if not all(inp[r, c] for r, c in gap) or any(inp[r, c] for r, c in spur):
            continue
        trial = inp.copy()
        for r, c in gap:
            trial[r, c] = False
        for r, c in spur:
            trial[r, c] = True
        try:
            recovered, evidence, erased = _recover(trial)
        except ValueError:
            continue
        if (np.array_equal(recovered, clean) and len(evidence) == done+1
                and all(erased[r, c] for r, c in spur)):
            inp, done = trial, done+1
            if done == count:
                break
    if done != count:
        return None
    # Optional detached two/three-cell strokes: never silently change geometry.
    for _ in range(int(rng.integers(3))):
        for _ in range(12):
            r, c = int(rng.integers(clean.shape[0])), int(rng.integers(clean.shape[1]))
            side, length = int(rng.choice([-1, 1])), int(rng.choice([2, 3]))
            cells = [(r+i, c+side*i) for i in range(length)]
            if not all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                       and not (inp[rr, cc] or clean[rr, cc]) for rr, cc in cells):
                continue
            trial = inp.copy()
            for rr, cc in cells:
                trial[rr, cc] = True
            try:
                recovered, evidence, _ = _recover(trial)
            except ValueError:
                continue
            if np.array_equal(recovered, clean) and len(evidence) == count:
                inp = trial
                break
    return inp


def _motifs(rng, s, k, witness):
    masks = []
    for _ in range(k):
        style = int(rng.integers(7))
        mask = np.zeros((s, s), bool)
        if style == 0:
            mask[:] = True
        elif style == 1:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
        elif style == 2:
            mask[int(rng.integers(s)), :] = True
        elif style == 3:
            np.fill_diagonal(mask, True)
        elif style == 4:
            rr, cc = np.indices((s, s))
            mask = (rr+cc) % 2 == int(rng.integers(2))
        else:
            mask = rng.random((s, s)) < rng.uniform(.2, .8)
        if not mask.any():
            mask.flat[int(rng.integers(s*s))] = True
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.integers(2):
            mask = np.fliplr(mask)
        masks.append(mask.copy())
    if witness:
        # A random orientation witness, rather than a compulsory named shape.
        for i in range(k):
            cells = rng.choice(s*s, size=int(rng.integers(1, s*s)), replace=False)
            masks[i] = np.zeros((s, s), bool)
            masks[i].flat[cells] = True
        if np.array_equal(masks[0], masks[1]):
            masks[1] = ~masks[0]
    elif k >= 2 and rng.random() < .2:
        masks[1] = masks[0].copy()
    return masks


def _pack(s, order, templates, targets, board):
    k, (h, w) = len(order), board.shape
    lw, top = k*(s+1)+1, s+5
    a = np.zeros((top+h+2, max(lw, w+2)), dtype=np.int8)
    a[0, :lw] = a[s+1, :lw] = a[s+3, :lw] = FRAME
    for j in range(k+1):
        a[:s+4, j*(s+1)] = FRAME
    for j, label in enumerate(order):
        c = 1+j*(s+1)
        a[1:s+1, c:c+s] = np.where(templates[label], label, BG)
        a[s+2, c+s//2] = targets[label]
    a[top, :w+2] = a[-1, :w+2] = FRAME
    a[top:, 0] = a[top:, w+1] = FRAME
    a[top+1:-1, 1:w+1] = board
    return a


def _diagnostics(p):
    clean, roots = p["clean"], p["roots"]
    winner, ties, unreachable, distances = _territory(clean, roots)
    without = clean.copy()
    for (r, c, dr, dc), _ in p["repairs"]:
        for i in (1, 2, 3):
            without[r+i*dr, c+i*dc] = False
    absent, _, _, _ = _territory(without, roots)
    manhattan, _, _, _ = _territory(clean, roots, "manhattan")
    connected_competition = (distances < INF).all(axis=0)
    all_gaps = clean.copy()
    for r, c, dr, dc in _gaps(clean):
        for i in (1, 2, 3):
            all_gaps[r+i*dr, c+i*dc] = True
    all_winner, _, _, _ = _territory(all_gaps, roots)
    result = _render(p, winner)
    return dict(winner=winner, ties=int(ties.sum()), unreachable=int(unreachable.sum()),
                global_repair=int(np.sum(without & (winner != absent))),
                identity_switch=int(np.sum(without & (winner >= 0) & (absent >= 0)
                                           & (winner != absent))),
                metric=int(np.sum(clean & connected_competition & (winner != manhattan))),
                intentional=len(_gaps(clean)),
                fill_all_wrong=not np.array_equal(result, _render(p, all_winner)),
                position_wrong=not np.array_equal(result, _render(p, winner, "position")),
                key_wrong=not np.array_equal(result, _render(p, winner, "unkeyed")),
                solid_wrong=not np.array_equal(result, _render(p, winner, "solid")))


def _make_pair(rng, s, k, h, w, repairs, witness, extra_mode="random"):
    for _ in range(320):
        clean = _network(rng, h, w)
        if witness or rng.random() < .65:
            options = _options(clean, with_spur=False)
            if not options:
                continue
            gap, _ = options[int(rng.integers(len(options)))]
            for r, c in gap:
                clean[r, c] = False
        try:
            recovered, _, _ = _recover(clean)
        except ValueError:
            continue
        if _thick(clean) or not np.array_equal(recovered, clean):
            continue
        inp = _corrupt(rng, clean, repairs)
        if inp is None:
            continue
        available = clean & inp
        components = _components(clean)
        largest = max(components, key=len)
        pool = [p for p in largest if available[p]]
        if len(pool) < k:
            continue
        labels = [int(x) for x in rng.choice(LABELS, size=k, replace=False)]
        for _ in range(28):
            # A minority of non-witness examples seed different components.
            use_pool = pool if witness or rng.random() < .8 else list(map(tuple, np.argwhere(available)))
            positions = [use_pool[int(i)] for i in rng.choice(len(use_pool), k, replace=False)]
            if witness and ((sum(positions[0])-sum(positions[1])) % 2):
                continue
            roots = {label: [position] for label, position in zip(labels, positions)}
            if not witness and (extra_mode == "force" or rng.random() < .3):
                used = set(positions)
                extra = [p for p in use_pool if p not in used]
                if extra:
                    roots[labels[int(rng.integers(k))]].append(extra[int(rng.integers(len(extra)))])
                elif extra_mode == "force":
                    continue
            board = np.where(inp, WIRE, BG).astype(np.int8)
            for label, points in roots.items():
                for r, c in points:
                    board[r, c] = label
            masks = _motifs(rng, s, k, witness)
            templates = dict(zip(labels, masks))
            targets = {label: int(x) for label, x in zip(labels, rng.choice(np.arange(1, 10), k, replace=False))}
            if not witness and rng.random() < .2:
                targets[labels[1]] = targets[labels[0]]
            if witness and any(targets[label] == label for label in labels):
                continue
            order = [int(x) for x in rng.permutation(labels)]
            if witness:
                seen_order = list(dict.fromkeys(int(x) for x in board.flat if int(x) in roots))
                order = seen_order[::-1]
            input_grid = _pack(s, order, templates, targets, board)
            p = _parse(input_grid)
            if not np.array_equal(p["clean"], clean):
                continue
            if witness:
                d = _diagnostics(p)
                if not (d["ties"] and d["global_repair"] >= 2 and d["identity_switch"]
                        and d["metric"]
                        and d["intentional"] and d["fill_all_wrong"]
                        and d["position_wrong"] and d["key_wrong"] and d["solid_wrong"]):
                    continue
            # Construction uses the pre-corruption graph, not solve's recovery.
            winner, _, _, _ = _territory(clean, roots)
            output = np.zeros((h*s, w*s), dtype=np.int8)
            for label in labels:
                output[np.kron(winner == label, templates[label])] = targets[label]
            return {"input": input_grid, "output": output}
    raise RuntimeError("bounded graph/witness sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime, s = int(rng.integers(3)), int(rng.choice([2, 3]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            witness = split == "train" and index == 0
            k = (2 if split == "train" else 3) if regime == 0 else int(rng.integers(2, 4))
            if witness:
                k = 2
            repairs = (1 if split == "train" else 2) if regime == 1 else int(rng.choice([1, 1, 2]))
            if regime == 2:
                h = int(rng.integers(8, 10)) if split == "train" else 10
                w = int(rng.integers(8, 10)) if split == "train" else int(rng.integers(9, 11))
            else:
                h, w = map(int, rng.integers(8, 11, size=2))
            for _ in range(8):
                extra_mode = "force" if split == "train" and index == 1 else "random"
                pair = _make_pair(rng, s, k, h, w, repairs, witness, extra_mode)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, scales = set(), [], set()
    repeated = {"train": False, "test": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            p = _parse(inp)
            repeated[split] |= any(len(points) > 1 for points in p["roots"].values())
            scales.add(p["s"])
            clean = p["clean"]
            assert not _thick(clean), "thick wire geometry"
            assert p["repairs"], "each example must expose real restoration"
            changed = clean & ~(p["board"] != BG)
            assert int(changed.sum()) == 3*len(p["repairs"])
            assert p["erased"].any()
            winner, ties, unreachable, _ = _territory(clean, p["roots"])
            expected = _render(p, winner)
            assert np.array_equal(out, expected), "incorrect/incomplete territories or stamps"
            assert out.shape == tuple(n*p["s"] for n in clean.shape)
            assert set(map(int, np.unique(out))) <= {BG, *p["targets"].values()}
            expected_ink = sum(int(np.sum(winner == label))*int(mask.sum())
                               for label, mask in p["templates"].items())
            assert np.count_nonzero(out) == expected_ink
            for label, points in p["roots"].items():
                assert all(winner[r, c] == label for r, c in points)
            empty_tiles = ties | unreachable | ~clean
            assert np.all(out[np.kron(empty_tiles, np.ones((p["s"], p["s"]), bool))] == BG)
            if split == "train":
                training.append(_diagnostics(p))
    assert len(scales) == 1, "motif scale is shared inside a puzzle"
    assert any(d["global_repair"] >= 2 for d in training), "no distant repair consequence"
    assert any(d["identity_switch"] for d in training), "repair never changes an existing identity"
    assert any(d["metric"] for d in training), "no evidence for wire rather than spatial distance"
    assert any(d["ties"] for d in training), "no neutral-tie demonstration"
    assert any(d["intentional"] and d["fill_all_wrong"] for d in training)
    assert any(d["position_wrong"] and d["key_wrong"] and d["solid_wrong"] for d in training)
    assert not repeated["test"] or repeated["train"], "undemonstrated repeated terminals"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; keep feature sets/hashes, never saved puzzle payloads."""
    shapes, signatures, maps, motifs, palettes = set(), set(), set(), set(), set()
    repairs, root_counts, sizes, cycles, unreachable = set(), set(), set(), set(), 0
    shifts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        local = {split: [] for split in ("train", "test")}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                p = _parse(pair["input"])
                clean = p["clean"]
                shapes.add((pair["input"].shape, pair["output"].shape))
                maps.add((clean.shape, clean.tobytes()))
                motifs.update((m.shape, m.tobytes()) for m in p["templates"].values())
                palettes.add((tuple(p["order"]), tuple(p["targets"].values())))
                repairs.add(len(p["repairs"]))
                root_counts.add(sum(map(len, p["roots"].values())))
                sizes.add(p["s"])
                edges = int(np.sum(clean[:-1] & clean[1:]) + np.sum(clean[:, :-1] & clean[:, 1:]))
                loops = edges-int(clean.sum())+len(_components(clean))
                cycles.add(loops)
                unreachable += _diagnostics(p)["unreachable"] > 0
                local[split].append((p["k"], len(p["repairs"]), clean.shape[0]))
        for axis, name in enumerate(("library", "repair_count", "map_height")):
            shifts[name] += min(p[axis] for p in local["test"]) > max(p[axis] for p in local["train"])
    assert len(signatures) == 42 and len(maps) >= 100
    assert len(shapes) >= 20 and len(motifs) >= 30 and len(palettes) >= 100
    assert sizes == {2, 3} and repairs >= {1, 2}
    assert len(root_counts) >= 3 and 0 in cycles and max(cycles) >= 1
    assert unreachable >= 5 and all(shifts[name] >= 1 for name in shifts)
    first = generate_puzzle(0)
    assert _signature(first) == _signature(generate_puzzle(0))
    assert _signature(first) != _signature(generate_puzzle(1))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(314159, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Small, hand-computable graph checks, including cooperative same-label roots.
    line = np.ones((1, 7), bool)
    winner, ties, _, _ = _territory(line, {1: [(0, 0)], 2: [(0, 6)]})
    assert winner.tolist() == [[1, 1, 1, -1, 2, 2, 2]] and ties.sum() == 1
    winner, ties, _, _ = _territory(line, {1: [(0, 0), (0, 6)]})
    assert np.all(winner == 1) and not ties.any()
    mask = np.zeros((7, 9), bool)
    mask[3, :] = True
    mask[3:, 0] = True
    mask[0, :2] = mask[0, 5:] = True  # Unsupported break remains.
    damaged = mask.copy()
    damaged[3, 3:6] = False
    damaged[4, 3] = damaged[5, 4] = True
    recovered, evidence, _ = _recover(damaged)
    assert np.array_equal(recovered, mask) and len(evidence) == 1
    # A complete hand-constructed 6x7 member, outside the sampler's size range.
    # Its three roots and asymmetric motifs use the same input-only interface.
    clean = np.zeros((6, 7), bool)
    clean[:, 0] = clean[2, :] = clean[2:, 6] = True
    board = np.where(clean, WIRE, BG).astype(np.int8)
    board[2, 2:5] = BG
    board[3, 2] = board[4, 3] = WIRE
    board[0, 0], board[5, 6], board[5, 0] = 1, 2, 3
    masks = {1: np.array([[1, 0], [0, 0]], bool),
             2: np.array([[0, 0], [0, 1]], bool),
             3: np.array([[0, 1], [1, 0]], bool)}
    query = _pack(2, [3, 1, 2], masks, {1: 9, 2: 8, 3: 7}, board)
    owners = np.full((6, 7), -1, dtype=np.int8)
    owners[:3, 0] = owners[2, :4] = 1
    owners[2, 4:] = owners[2:, 6] = 2
    owners[3:, 0] = 3
    expected = np.zeros((12, 14), dtype=np.int8)
    for label, color in ((1, 9), (2, 8), (3, 7)):
        expected[np.kron(owners == label, masks[label])] = color
    assert np.array_equal(solve(query), expected)
    # A wrong tile, an omitted repair consequence, and a shuffled key are labels
    # that reconstruction must reject. No assertion checks only edited cells.
    for mutation in ("pixel", "omit_repair", "key"):
        bad = {split: [{role: value.copy() for role, value in pair.items()}
                       for pair in first[split]] for split in ("train", "test")}
        pair = bad["train"][0]
        p = _parse(pair["input"])
        if mutation == "pixel":
            pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
        elif mutation == "omit_repair":
            without = p["clean"].copy()
            for (r, c, dr, dc), _ in p["repairs"]:
                for i in (1, 2, 3):
                    without[r+i*dr, c+i*dc] = False
            pair["output"] = _render(p, _territory(without, p["roots"])[0])
        else:
            c0, c1 = 1+p["s"]//2, 1+(p["s"]+1)+p["s"]//2
            row = p["s"]+2
            pair["input"][row, c0], pair["input"][row, c1] = (
                pair["input"][row, c1], pair["input"][row, c0])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"{mutation} mutation accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), graph_geometries=len(maps),
                shape_pairs=len(shapes), motif_bitmaps=len(motifs), palettes=len(palettes),
                repair_counts=sorted(repairs), terminal_counts=sorted(root_counts),
                cycle_counts=sorted(cycles), pairs_with_unreachable_wire=unreachable,
                observed_shifts=dict(shifts))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
