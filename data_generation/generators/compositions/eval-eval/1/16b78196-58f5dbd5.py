"""Color-requested negative molds, assembled by translation-only exact cover.

Only Python's standard library and NumPy are used. Generation metadata never
enters the decoder or packing search. Rectangular mold rims expose the register;
the assembly board is the unique multicolored framed component.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    """Four-connected components of a Boolean mask, irrespective of color."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    components = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        components.append(np.array(sorted(cells), dtype=int))
    return components


def _rim(shape):
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _connected(mask):
    return bool(mask.any()) and len(_components(mask)) == 1


def _orientation_candidates(masks):
    candidates = set(range(8))
    for mask in masks:
        transforms = [np.rot90(mask, i) for i in range(4)]+[
            np.fliplr(np.rot90(mask, i)) for i in range(4)]
        candidates &= {i for i, other in enumerate(transforms)
                       if np.array_equal(mask, other)}
    return candidates


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    boards, molds, boxes = [], {}, []
    for cells in _components(a != bg):
        lo, hi = cells.min(axis=0), cells.max(axis=0)+1
        y, x, bottom, right = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
        window = a[y:bottom, x:right].copy()
        assert min(window.shape) >= 3, "unframed component"
        rim = _rim(window.shape)
        assert np.all(window[rim] != bg), "incomplete rectangular rim"
        colors = set(map(int, np.unique(window[window != bg])))
        box = (y, x, bottom, right)
        boxes.append(box)
        if len(colors) > 1:
            boards.append((window, box))
        else:
            color = next(iter(colors))
            assert color not in molds, "duplicate mold color"
            shape = window[1:-1, 1:-1] == bg
            assert _connected(shape), "cast must be a nonempty connected piece"
            assert all((shape[0].any(), shape[-1].any(),
                        shape[:, 0].any(), shape[:, -1].any())), "cast register is tight"
            molds[color] = {"shape": shape, "ink": ~shape, "box": box}
    assert len(boards) == 1, "expected one multicolored assembly board"
    board, box = boards[0]
    rim = _rim(board.shape)
    counts = np.bincount(board[rim], minlength=10)
    structure = int(counts.argmax())
    assert counts[structure] > np.count_nonzero(rim)//2
    interior = board[1:-1, 1:-1]
    assert np.all((interior == bg) | (interior == structure))
    assert np.any(interior == structure), "board needs fixed interior material"
    requests = sorted(set(map(int, board[rim]))-{structure})
    assert 2 <= len(requests) <= 6
    assert bg not in requests and structure not in molds
    assert set(requests) <= set(molds), "a requested color has no mold"
    corners = board[np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])]
    assert np.all(corners == structure)
    request_cells = np.argwhere(rim & (board != structure))
    assert len(request_cells) == len(requests), "exactly one dot per request"
    for i, p in enumerate(request_cells):
        for q in request_cells[i+1:]:
            assert np.max(np.abs(p-q)) > 1, "inventory dots must be isolated"
    # Separated bounding rectangles keep the complete board and molds visible.
    for i, first in enumerate(boxes):
        for second in boxes[i+1:]:
            assert (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1])
    pieces = {c: molds[c]["shape"] for c in requests}
    return {"input": a, "background": bg, "structure": structure,
            "board": board, "board_box": box, "pieces": pieces,
            "molds": molds, "requests": requests,
            "dot_order": tuple(int(board[tuple(p)]) for p in request_cells),
            "mold_order": tuple(sorted(requests, key=lambda c: molds[c]["box"][:2])),
            "distractors": sorted(set(molds)-set(requests))}


def _pack(board, background, pieces, limit=2):
    """All translations of arbitrary registered masks, then global exact cover.

    Unique colors make distinct assignments distinct colored answers. The search
    stops at `limit` answers; two suffice to detect ambiguity, not to count it.
    No chain, orientation, size ordering, source layout, or generation label is
    assumed. Minimum-remaining-values branching keeps the bounded cases small.
    """
    cavity = board[1:-1, 1:-1] == background
    h, w = cavity.shape
    if sum(int(m.sum()) for m in pieces.values()) != int(cavity.sum()):
        return [], {"placements": 0, "nodes": 0}
    colors = sorted(pieces)
    cell_ids = {tuple(map(int, p)): i for i, p in enumerate(np.argwhere(cavity))}
    full = (1 << len(cell_ids))-1
    by_cell = [[] for _ in cell_ids]
    candidates = []
    for index, color in enumerate(colors):
        mask = pieces[color]
        cells = np.argwhere(mask)
        ph, pw = mask.shape
        for y in range(h-ph+1):
            for x in range(w-pw+1):
                destination = cells+np.array([y, x])
                if not np.all(cavity[destination[:, 0], destination[:, 1]]):
                    continue
                ids = [cell_ids[tuple(map(int, p))] for p in destination]
                bits = sum(1 << k for k in ids)
                item = (index, bits, color, y, x, destination)
                candidates.append(item)
                for k in ids:
                    by_cell[k].append(item)
    answers, nodes = [], 0

    def visit(occupied, used, chosen):
        nonlocal nodes
        nodes += 1
        if len(answers) >= limit:
            return
        if occupied == full:
            if used != (1 << len(colors))-1:
                return
            out = board.copy()
            placements = {}
            for _, _, color, y, x, cells in chosen:
                out[cells[:, 0]+1, cells[:, 1]+1] = color
                placements[color] = (y, x)
            answers.append((out, placements))
            return
        best = None
        for k, items in enumerate(by_cell):
            if occupied & (1 << k):
                continue
            available = [v for v in items
                         if not (used & (1 << v[0]) or occupied & v[1])]
            if not available:
                return
            if best is None or len(available) < len(best):
                best = available
                if len(best) == 1:
                    break
        for v in best:
            visit(occupied | v[1], used | (1 << v[0]), chosen+[v])
            if len(answers) >= limit:
                return

    visit(0, 0, [])
    return answers, {"placements": len(candidates), "nodes": nodes}


def _solution(input_grid):
    info = _decode(input_grid)
    answers, search = _pack(info["board"], info["background"], info["pieces"])
    assert len(answers) == 1, "board must have one distinct colored exact cover"
    output, placements = answers[0]
    info.update(search=search, destinations=placements)
    info["destination_order"] = tuple(sorted(placements, key=lambda c: placements[c]))
    return output.copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the complete board from the input alone."""
    return _solution(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _anchors(rng, h, w, mode):
    """Jagged edges, opposite edges, or a through-divider; all attach to rim."""
    a = np.zeros((h, w), dtype=bool)
    if mode == 0:
        depths = rng.integers(0, 3, size=w)
        depths[int(rng.integers(w))] = 2
        for c, d in enumerate(depths):
            a[:d, c] = True
    elif mode == 1:
        for c in range(w):
            a[:int(rng.integers(0, 3)), c] = True
            d = int(rng.integers(0, 3))
            if d:
                a[h-d:, c] = True
    else:
        middle = int(rng.integers(2, w-2))
        for r in range(h):
            a[r, middle-int(rng.integers(0, 2)):middle+1+int(rng.integers(0, 2))] = True
        if rng.random() < .5:
            a[0, :int(rng.integers(1, middle+1))] = True
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _partition(rng, free, n):
    """Balanced random growth creates genuinely two-dimensional jigsaw keys."""
    components = _components(free)
    if len(components) > n:
        return None
    labels = np.full(free.shape, -2, dtype=int)
    labels[free] = -1
    seeds = [tuple(map(int, cells[int(rng.integers(len(cells)))])) for cells in components]
    remaining = [tuple(map(int, p)) for p in np.argwhere(free) if tuple(map(int, p)) not in seeds]
    # Spread additional seeds to avoid tiny regions while retaining randomness.
    while len(seeds) < n:
        if not remaining:
            return None
        distances = [min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in seeds) for p in remaining]
        choices = [p for p, d in zip(remaining, distances) if d >= max(distances)-1]
        p = _pick(rng, choices)
        seeds.append(p)
        remaining.remove(p)
    frontiers = [set() for _ in seeds]
    sizes = np.ones(n, dtype=int)

    def neighbors(p):
        r, c = p
        return [(y, x) for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                if 0 <= y < free.shape[0] and 0 <= x < free.shape[1]]

    for i, p in enumerate(seeds):
        labels[p] = i
    for i, p in enumerate(seeds):
        frontiers[i].update(q for q in neighbors(p) if labels[q] == -1)
    while np.any(labels == -1):
        eligible = [i for i in range(n) if frontiers[i]]
        if not eligible:
            return None
        weights = np.array([1/(sizes[i]**float(.7)) for i in eligible])
        i = int(rng.choice(eligible, p=weights/weights.sum()))
        p = _pick(rng, sorted(frontiers[i]))
        labels[p] = i
        sizes[i] += 1
        for frontier in frontiers:
            frontier.discard(p)
        frontiers[i].update(q for q in neighbors(p) if labels[q] == -1)
    return labels


def _make_mold(mask, color, bg):
    mold = np.full((mask.shape[0]+2, mask.shape[1]+2), color, dtype=np.int8)
    mold[1:-1, 1:-1][mask] = bg
    return mold


def _layout(rng, blocks, bg):
    """Place independent rectangles with clearance; positions encode no order."""
    for _ in range(36):
        h, w = 30, 30
        occupied = np.zeros((h, w), dtype=bool)
        occupied[0] = occupied[-1] = True
        occupied[:, 0] = occupied[:, -1] = True
        grid = np.full((h, w), bg, dtype=np.int8)
        boxes = []
        order = sorted(range(len(blocks)),
                       key=lambda i: -(blocks[i].size+rng.uniform(0, 12)))
        success = True
        for i in order:
            block = blocks[i]
            bh, bw = block.shape
            choices = [(y, x) for y in range(1, h-bh)
                       for x in range(1, w-bw)
                       if not occupied[y:y+bh, x:x+bw].any()]
            if not choices:
                success = False
                break
            y, x = _pick(rng, choices)
            grid[y:y+bh, x:x+bw] = block
            occupied[y-1:y+bh+1, x-1:x+bw+1] = True
            boxes.append((y, x, y+bh, x+bw))
        if success:
            y0 = min(b[0] for b in boxes)-1
            x0 = min(b[1] for b in boxes)-1
            y1 = max(b[2] for b in boxes)+1
            x1 = max(b[3] for b in boxes)+1
            grid = grid[y0:y1, x0:x1].copy()
            padding = []
            for d in grid.shape:
                extra = int(rng.integers(31-d))
                leading = int(rng.integers(extra+1))
                padding.append((leading, extra-leading))
            return np.pad(grid, padding, constant_values=bg)
    return None


def _random_distractor(rng):
    """Connected small hole shapes, with connected ink and a visible rim."""
    for _ in range(64):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        m = rng.random((h, w)) < rng.uniform(.35, .8)
        cells = np.argwhere(m)
        if not len(cells):
            continue
        lo, hi = cells.min(axis=0), cells.max(axis=0)+1
        m = m[lo[0]:hi[0], lo[1]:hi[1]]
        if min(m.shape) >= 2 and _connected(m) and _connected(_make_mold(m, 1, 0) != 0):
            return m.copy()
    return np.array([[True, True, False], [False, True, True]], dtype=bool)


def _make_pair(rng, n, mode, rotate, role_colors, witness, distractor_count):
    bg, structure = role_colors
    for _ in range(180):
        h = int(rng.integers(5+n//2, 9+n//2))
        w = int(rng.integers(5+n//2, 10+n//2))
        anchors = _anchors(rng, h, w, mode)
        if not anchors.any():
            continue
        labels = _partition(rng, ~anchors, n)
        if labels is None:
            continue
        colors = [int(c) for c in rng.choice(
            [c for c in range(10) if c not in (bg, structure)],
            n+distractor_count, replace=False)]
        board = np.full((h+2, w+2), structure, dtype=np.int8)
        board[1:-1, 1:-1] = np.where(anchors, structure, bg)
        output = board.copy()
        shapes = {}
        good = True
        for i, color in enumerate(colors[:n]):
            cells = np.argwhere(labels == i)
            lo, hi = cells.min(axis=0), cells.max(axis=0)+1
            mask = labels[lo[0]:hi[0], lo[1]:hi[1]] == i
            if (len(cells) < 4 or min(mask.shape) < 2
                    or max(mask.shape) > 11
                    or not _connected(_make_mold(mask, color, bg) != bg)):
                good = False
                break
            shapes[color] = mask.copy()
            output[1:-1, 1:-1][labels == i] = color
        if not good:
            continue
        if witness and (sum(not m.all() for m in shapes.values()) < 2
                        or all(2*m.sum() == m.size for m in shapes.values())
                        or _orientation_candidates(shapes.values()) != {0}):
            continue
        # Isolated non-corner inventory dots, in an independent random order.
        available = [tuple(map(int, p)) for p in np.argwhere(_rim(board.shape))
                     if tuple(p) not in ((0, 0), (0, w+1), (h+1, 0), (h+1, w+1))]
        for color in rng.permutation(colors[:n]):
            if not available:
                good = False
                break
            p = _pick(rng, available)
            board[p] = output[p] = int(color)
            available = [q for q in available if max(abs(q[0]-p[0]), abs(q[1]-p[1])) > 1]
        if not good:
            continue
        # Whole-scene rotations preserve relative stencil/board orientation.
        board, output = np.rot90(board, rotate).copy(), np.rot90(output, rotate).copy()
        shapes = {c: np.rot90(m, rotate).copy() for c, m in shapes.items()}
        answers, _ = _pack(board, bg, shapes)
        if len(answers) != 1 or not np.array_equal(answers[0][0], output):
            continue
        distractors = {c: _random_distractor(rng) for c in colors[n:]}
        if witness:
            # A decoy with the exact shape of a requested mold makes selection
            # depend on the inventory color, even when shape also looks useful.
            c = colors[n]
            distractors[c] = shapes[colors[0]].copy()
        molds = {**shapes, **distractors}
        blocks = [board]+[_make_mold(m, c, bg) for c, m in molds.items()]
        inp = _layout(rng, blocks, bg)
        if inp is None:
            continue
        predicted, info = _solution(inp)
        if witness and (info["mold_order"] == info["destination_order"]
                        or info["dot_order"] == info["destination_order"]):
            continue
        assert np.array_equal(predicted, output)
        return {"input": inp, "output": output}
    raise RuntimeError("could not construct a uniquely fitting mosaic in bounded retries")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    # These preferences govern sampling only. All conventions are input-visible.
    stable_roles = rng.random() < .25
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    rotate_base = int(rng.integers(4))
    orientation_transfer = bool(rng.integers(2))
    training_ceiling = int(rng.integers(3, 5))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt+ne):
        training = index < nt
        witness = index < 2
        if training:
            n = 2 if index == 0 else (3 if index == 1 else int(rng.integers(2, training_ceiling+1)))
            mode = index % 3 if index < 2 else int(rng.integers(3))
        else:
            n = int(rng.integers(training_ceiling+1, 7))
            mode = int(rng.integers(3))
        rotate = ((rotate_base+(0 if training else 1)) % 4
                  if orientation_transfer else int(rng.integers(4)))
        current_roles = roles if stable_roles else tuple(map(int, rng.choice(10, 2, replace=False)))
        d = int(rng.integers(1 if witness else 0, min(2, 8-n)+1))
        for _ in range(8):
            pair = _make_pair(rng, n, mode, rotate, current_roles, witness, d)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not create a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), 2, 10, "num_train")
    _count(len(puzzle["test"]), 1, 4, "num_test")
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _solution(inp)
            assert np.array_equal(out, expected), "wrong cast, placement, or crop"
            board, bg, structure = info["board"], info["background"], info["structure"]
            assert out.shape == board.shape
            assert np.array_equal(out[_rim(out.shape)], board[_rim(board.shape)])
            assert np.array_equal(out == structure, board == structure)
            assert np.all(out[1:-1, 1:-1] != bg), "unfilled board cell"
            for color, mask in info["pieces"].items():
                target = out[1:-1, 1:-1] == color
                cells = np.argwhere(target)
                lo, hi = cells.min(axis=0), cells.max(axis=0)+1
                assert np.array_equal(target[lo[0]:hi[0], lo[1]:hi[1]], mask)
                assert int(target.sum()) == int(mask.sum())
                assert np.all(board[1:-1, 1:-1][target] == bg)
            assert not set(info["distractors"]) & set(map(int, np.unique(out)))
            # Using ink instead of empty cells must not reproduce the answer.
            ink = {c: info["molds"][c]["ink"] for c in info["requests"]}
            wrong, _ = _pack(board, bg, ink)
            assert not any(np.array_equal(v[0], out) for v in wrong)
            if split == "train":
                train_infos.append(info)
    # Curriculum evidence, not a condition on solve(): the first two demos
    # demonstrate shape inversion, consequential color selection, and source
    # position independence, including a same-shape different-color decoy.
    for info in train_infos[:2]:
        assert len(info["distractors"]) >= 1
        assert sum(not m.all() for m in info["pieces"].values()) >= 2
        assert any(2*m.sum() != m.size for m in info["pieces"].values())
        assert info["mold_order"] != info["destination_order"]
        assert info["dot_order"] != info["destination_order"]
        assert any(np.array_equal(info["molds"][d]["shape"], m)
                   for d in info["distractors"] for m in info["pieces"].values())
        all_molds = {c: m["shape"] for c, m in info["molds"].items()}
        assert not _pack(info["board"], info["background"], all_molds)[0]
    assert _orientation_candidates(
        m for info in train_infos[:2] for m in info["pieces"].values()) == {0}


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded diversity audit, genuine geometric invariants, and regressions."""
    dimensions, outputs, inventories, backgrounds, anchor_counts = set(), set(), set(), set(), set()
    normalized, fingerprints = set(), set()
    nonconvex = disconnected_cavities = rectangular = decoys = local_choices = 0
    changed_roles = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        roles = set()
        for pair in puzzle["train"]+puzzle["test"]:
            total += 1
            out, info = _solution(pair["input"])
            assert np.array_equal(out, pair["output"])
            dimensions.add(pair["input"].shape)
            outputs.add(out.shape)
            inventories.add(len(info["pieces"]))
            backgrounds.add(info["background"])
            roles.add((info["background"], info["structure"]))
            anchor_counts.add(int((info["board"][1:-1, 1:-1] == info["structure"]).sum()))
            disconnected_cavities += len(_components(info["board"][1:-1, 1:-1] == info["background"])) > 1
            local_choices += info["search"]["placements"] > len(info["pieces"])
            decoys += len(info["distractors"])
            for m in info["pieces"].values():
                rectangular += bool(m.all())
                # A nonconvex row or column disproves a universal profile-chain
                # assumption; the solver still checks arbitrary occupied masks.
                nonconvex += any(len(np.flatnonzero(row)) > 1 and
                                 np.ptp(np.flatnonzero(row))+1 != row.sum()
                                 for row in list(m)+list(m.T))
            palette = {}
            values = tuple(palette.setdefault(int(c), len(palette)) for c in pair["input"].flat)
            normalized.add((pair["input"].shape, values))
        changed_roles += len(roles) > 1
    assert len(fingerprints) == 42 and len(normalized) >= .98*total
    assert len(dimensions) >= 12 and len(outputs) >= 20
    assert inventories == {2, 3, 4, 5, 6} and backgrounds == set(range(10))
    assert len(anchor_counts) >= 20 and disconnected_cavities >= 10
    assert nonconvex >= 40 and decoys >= 80 and local_choices >= 20
    assert 10 <= changed_roles < 42
    p = generate_puzzle(123, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(123, 2, 1))
    assert _fingerprint(p) != _fingerprint(generate_puzzle(124, 2, 1))
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        q = generate_puzzle(987+a+b, a, b)
        assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.2), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Generic exact-cover fixtures explicitly exercise ambiguity and boundaries.
    b = np.full((4, 6), 8, dtype=np.int8)
    b[1:-1, 1:-1] = 0
    halves = {2: np.ones((2, 2), bool), 3: np.ones((2, 2), bool)}
    assert len(_pack(b, 0, halves)[0]) == 2
    assert not _pack(b, 0, {2: np.ones((3, 3), bool)})[0]
    # A complete tiny family member outside the automatic sampler: a one-column
    # key, an ink-free rectangular mold, and a 3x3 board interior. Registration
    # and input-only solving must not depend on the sampled size restrictions.
    board = np.full((5, 5), 8, dtype=np.int8)
    board[1:4, 1:3] = 0
    board[0, 1], board[4, 3] = 2, 3
    l_key = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    line = np.ones((2, 1), dtype=bool)
    fixture = np.zeros((14, 14), dtype=np.int8)
    fixture[1:6, 1:6] = board
    fixture[1:6, 8:12] = _make_mold(l_key, 2, 0)
    fixture[8:12, 1:4] = _make_mold(line, 3, 0)
    expected = board.copy()
    expected[1:4, 1:3][l_key] = 2
    expected[1:3, 2] = 3
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turns)),
                              np.rot90(expected, turns))
    # Reposition all outside molds by reflecting the canvas, while restoring the
    # board and each mold's displayed orientation: only layout changes.
    inp = p["train"][0]["input"]
    info = _decode(inp)
    relocated = np.full_like(inp, info["background"])
    for box in [info["board_box"]]+[m["box"] for m in info["molds"].values()]:
        y, x, bottom, right = box
        relocated[y:bottom, inp.shape[1]-right:inp.shape[1]-x] = inp[y:bottom, x:right]
    assert np.array_equal(solve(relocated), p["train"][0]["output"])
    broken = copy.deepcopy(p)
    broken["test"][0]["output"][1, 1] = (int(broken["test"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted damaged mosaic")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(dimensions), "output_shapes": len(outputs),
            "normalized_inputs": len(normalized), "inventories": sorted(inventories),
            "backgrounds": sorted(backgrounds), "fixed_material_counts": len(anchor_counts),
            "nonconvex_pieces": nonconvex, "disconnected_cavities": disconnected_cavities,
            "rectangular_pieces": rectangular, "decoy_molds": decoys,
            "pairs_with_extra_local_placements": local_choices,
            "puzzles_with_changing_roles": changed_roles}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
