"""Voting-stencil negatives: connected-region votes, spatial quorum, microtiles.

Only standard-library and NumPy dependencies. No files, examples, or generation
state are consulted by solve. All diagnostics in self_test are memory bounded.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 992471232, 951391612, 496208391, 875681508, 226641991,
              28312283, 434987234, 841304144, 309522426, 511099527,
              423881958, 881309344, 113191970, 835610852, 803694275,
              567884375, 829150042, 452107679, 656933131, 849139094,
              744965720, 428323823, 911329863, 66575709, 612168870,
              876273220, 613646721, 260131866, 27324316, 764556333,
              407448964, 891816749, 72676360, 961741685, 433765648,
              126025888, 270257486, 979199045, 44409796, 537943152,
              566706397)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            p = todo.pop()
            cells.append(p)
            for dr, dc in DIRS:
                x, y = p[0] + dr, p[1] + dc
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    todo.append((x, y))
        result.append(cells)
    return result


def _region_labels(board, background, signals):
    labels = np.full(board.shape, -1, np.int8)
    areas = np.zeros(board.shape, np.int16)
    regions = _components(board != background)
    assert regions
    majorities = []
    for cells in regions:
        counts = [sum(board[p] == s for p in cells) for s in signals]
        assert len(cells) >= 3 and min(counts) > 0 and counts[0] != counts[1]
        majority = signals[int(counts[1] > counts[0])]
        majorities.append(majority)
        for p in cells:
            labels[p], areas[p] = majority, len(cells)
    return labels, areas, regions, majorities


def _decode_as(grid, background, frame):
    """Decode explicit frame components, never layout slots or pair ordinals."""
    stencil_records, tiles = [], []
    explained = grid == background
    for cells in _components(grid == frame):
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        h, w = max(rr) - r + 1, max(cc) - c + 1
        box = grid[r:r + h, c:c + w]
        assert min(h, w) >= 4
        assert all(np.all(edge == frame) for edge in
                   (box[0], box[-1], box[:, 0], box[:, -1]))
        if h == w:
            n = h - 2
            assert 2 <= n <= 5
            tile = box[1:-1, 1:-1].copy()
            assert frame not in tile and background not in tile
            assert len(np.unique(tile)) == 2
            tiles.append((tile, (r, c, h, w)))
        else:
            m = w - 2
            assert 3 <= m <= 7 and h == m + 6
            board = box[1:m + 1, 1:m + 1].copy()
            assert not np.any(board == frame)
            lower = box[m + 2:m + 5]
            points = np.argwhere(lower != frame)
            assert len(points) == 9
            lo, hi = points.min(0), points.max(0) + 1
            assert tuple(hi - lo) == (3, 3) and lo[0] == 0
            q = int(lo[1])
            assert 1 <= q <= m - 2
            key = lower[:, q:q + 3]
            marks = set(map(tuple, np.argwhere(key != background)))
            assert len(marks) == 2 and (1, 1) in marks
            neighbor = next(iter(marks - {(1, 1)}))
            delta = neighbor[0] - 1, neighbor[1] - 1
            assert delta in DIRS and len(np.unique(key[key != background])) == 1
            assert int(key[1, 1]) not in (background, frame)
            turn = DIRS.index(delta)
            expected = np.full(box.shape, frame, np.int8)
            expected[1:m + 1, 1:m + 1] = board
            expected[m + 2:m + 5, q:q + 3] = key
            assert np.array_equal(expected, box)
            stencil_records.append(dict(board=board, turn=turn, box=(r, c, h, w),
                                        key_origin=(r + m + 2, c + q)))
        explained[r:r + h, c:c + w] = True
    assert len(tiles) == 1 and 2 <= len(stencil_records) <= 5
    assert explained.all(), "unframed content or stray pixels"
    tile, tile_box = tiles[0]
    signals = tuple(map(int, np.unique(tile)))
    assert set(map(int, np.unique(grid))) == {background, frame, *signals}
    counts = [int(np.sum(tile == s)) for s in signals]
    assert counts[0] != counts[1]
    tile_majority = signals[int(counts[1] > counts[0])]
    m = stencil_records[0]["board"].shape[0]
    assert m * len(tile) <= 30
    for rec in stencil_records:
        board = rec["board"]
        assert board.shape == (m, m)
        assert set(map(int, np.unique(board))) <= {background, *signals}
        labels, areas, regions, majorities = _region_labels(board, background, signals)
        rec.update(labels=labels, areas=areas, regions=regions, majorities=majorities)
    return dict(background=background, frame=frame, tile=tile, tile_box=tile_box,
                tile_majority=tile_majority, signals=signals, m=m, sheets=stencil_records)


def _decode(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    assert np.all(grid[0] == background) and np.all(grid[-1] == background)
    interpretations = []
    for frame in map(int, np.unique(grid)):
        if frame == background:
            continue
        try:
            interpretations.append(_decode_as(grid, background, frame))
        except (AssertionError, IndexError, ValueError):
            pass
    assert len(interpretations) == 1, "expected one explicit frame interpretation"
    return interpretations[0]


def _ballots(info, behavior="normal", convention=(1, 0)):
    signals, m = info["signals"], info["m"]
    votes = np.zeros((2, m, m), np.int32)
    population = np.zeros((m, m), np.int16)
    for rec in info["sheets"]:
        labels = rec["labels"].copy()
        weights = np.ones((m, m), np.int16)
        if behavior == "raw_pixels":
            labels = np.where(rec["board"] == info["background"], -1, rec["board"])
        elif behavior == "whole_sheet":
            counts = [int(np.sum(rec["board"] == s)) for s in signals]
            winner = -1 if counts[0] == counts[1] else signals[int(counts[1] > counts[0])]
            labels[labels >= 0] = winner
        elif behavior == "area_weighted":
            weights = rec["areas"]
        turn = 0 if behavior == "no_rotation" else (convention[0] * rec["turn"] + convention[1]) % 4
        labels = np.rot90(labels, -turn)
        weights = np.rot90(weights, -turn)
        population += labels >= 0
        for j, signal in enumerate(signals):
            votes[j] += (labels == signal) * weights
    verdict = np.full((m, m), -1, np.int8)
    verdict[votes[0] > votes[1]] = signals[0]
    verdict[votes[1] > votes[0]] = signals[1]
    return verdict, votes, population


def _expand(info, verdict, behavior="normal"):
    n, m = len(info["tile"]), info["m"]
    rr, cc = np.indices((m * n, m * n))
    v = verdict[rr // n, cc // n]
    majority_sites = info["tile"] == info["tile_majority"]
    if behavior == "rotate_tile":
        majority_sites = np.rot90(majority_sites, -1)
    stencil = majority_sites[rr % n, cc % n]
    if behavior == "no_exchange":
        stencil = ~stencil
    a, b = info["signals"]
    result = np.where(stencil, a + b - v, v)
    result[v < 0] = info["background"]
    return result.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer components and controls from input pixels and print spatial verdicts."""
    info = _decode(input_grid)
    return _expand(info, _ballots(info)[0])


def _inverse_prediction(info):
    """Generator-owned audit: inverse turns and individual output pixel lookup."""
    m, n = info["m"], len(info["tile"])
    a, b = info["signals"]
    verdict = np.full((m, m), -1, np.int8)
    for r in range(m):
        for c in range(m):
            tally = Counter()
            for rec in info["sheets"]:
                x, y = r, c
                for _ in range(rec["turn"]):
                    x, y = m - 1 - y, x
                label = int(rec["labels"][x, y])
                if label >= 0:
                    tally[label] += 1
            if tally[a] != tally[b]:
                verdict[r, c] = a if tally[a] > tally[b] else b
    out = np.full((m * n, m * n), info["background"], np.int8)
    for r in range(len(out)):
        for c in range(len(out)):
            winner = int(verdict[r // n, c // n])
            if winner >= 0:
                dominant_site = info["tile"][r % n, c % n] == info["tile_majority"]
                out[r, c] = a + b - winner if dominant_site else winner
    return out


def _mask(rng, m, force_split=False):
    """Rectangles, bars, polyominoes, contours, branches, and density masks."""
    kind = int(rng.integers(6))
    mask = np.zeros((m, m), bool)
    if force_split:
        # A visible empty line separates two patches of independently chosen sizes.
        cut = int(rng.integers(1, m - 1))
        for start, end in ((0, cut), (cut + 1, m)):
            width = int(rng.integers(3, m + 1))
            col = int(rng.integers(m - width + 1))
            mask[start:end, col:col + width] = True
    elif kind == 0:
        for _ in range(int(rng.integers(1, 5))):
            h, w = map(int, rng.integers(1, m + 1, 2))
            if h * w < 3:
                continue
            r, c = int(rng.integers(m - h + 1)), int(rng.integers(m - w + 1))
            mask[r:r + h, c:c + w] = True
    elif kind == 1:
        mask = rng.random((m, m)) < rng.uniform(.4, .82)
    elif kind == 2:
        r, c = int(rng.integers(m)), int(rng.integers(m))
        mask[r, c] = True
        for _ in range(int(rng.integers(m * 2, m * m * 3))):
            dr, dc = DIRS[int(rng.integers(4))]
            r, c = int(np.clip(r + dr, 0, m - 1)), int(np.clip(c + dc, 0, m - 1))
            mask[r, c] = True
    elif kind == 3:
        h, w = int(rng.integers(3, m + 1)), int(rng.integers(3, m + 1))
        r, c = int(rng.integers(m - h + 1)), int(rng.integers(m - w + 1))
        mask[r:r + h, c] = mask[r:r + h, c + w - 1] = True
        mask[r, c:c + w] = mask[r + h - 1, c:c + w] = True
    elif kind == 4:
        for r in range(0, m, 2):
            start = int(rng.integers(max(1, m - 2)))
            end = int(rng.integers(start + 3, m + 1))
            mask[r, start:end] = True
    else:
        r, c = int(rng.integers(m)), int(rng.integers(m))
        mask[r, :] = mask[:, c] = True
        mask |= rng.random((m, m)) < .18
    mask = np.rot90(mask, int(rng.integers(4))).copy()
    for cells in _components(mask):
        if len(cells) < 3:
            for p in cells:
                mask[p] = False
    if not mask.any():
        mask[int(rng.integers(m)), :] = True
    return mask


def _source(rng, m, background, signals, force_split):
    mask = _mask(rng, m, force_split)
    regions = _components(mask)
    labels = np.full((m, m), -1, np.int8)
    board = np.full((m, m), background, np.int8)
    majorities = list(map(int, rng.integers(2, size=len(regions))))
    if force_split:
        assert len(regions) >= 2
        majorities[:2] = list(map(int, rng.permutation(2)))
    for cells, majority_index in zip(regions, majorities):
        majority, minority = signals[majority_index], signals[1 - majority_index]
        count = len(cells)
        nminor = int(rng.integers(1, (count - 1) // 2 + 1))
        order = rng.permutation(count) if rng.random() < .65 else np.arange(count)
        minor_indices = set(map(int, order[:nminor]))
        for j, p in enumerate(cells):
            board[p] = minority if j in minor_indices else majority
            labels[p] = majority
    return board, labels


def _tile(rng, n, signals):
    minority_count = int(rng.integers(1, (n * n - 1) // 2 + 1))
    bits = np.zeros(n * n, bool)
    for _ in range(64):
        bits[:] = False
        bits[rng.choice(n * n, minority_count, replace=False)] = True
        square = bits.reshape(n, n)
        if all(not np.array_equal(square, np.rot90(square, k)) for k in (1, 2, 3)):
            break
    else:
        square = np.zeros((n, n), bool)
        square[0, 0] = True
    majority = signals[int(rng.integers(2))]
    return np.where(square, sum(signals) - majority, majority).astype(np.int8), majority


def _card(board, turn, background, frame, marker, key_offset):
    m = len(board)
    card = np.full((m + 6, m + 2), frame, np.int8)
    card[1:m + 1, 1:m + 1] = board
    row, col = m + 2, 1 + key_offset
    card[row:row + 3, col:col + 3] = background
    card[row + 1, col + 1] = marker
    dr, dc = DIRS[turn]
    card[row + 1 + dr, col + 1 + dc] = marker
    return card


def _pack(rng, cards, background):
    """Variable object order, gaps, margins, and prototype position."""
    count = len(cards)
    cols = 3 if count >= 5 else int(rng.choice([2, 3]))
    rows = (count + cols - 1) // cols
    slot_h = max(a.shape[0] for a in cards)
    slot_w = max(a.shape[1] for a in cards)
    gap_y = int(rng.integers(1, min(3, (28 - rows * slot_h) // max(1, rows - 1)) + 1)) if rows > 1 else 1
    gap_x = int(rng.integers(1, min(3, (30 - cols * slot_w) // max(1, cols - 1)) + 1))
    base_h, base_w = rows * slot_h + (rows - 1) * gap_y, cols * slot_w + (cols - 1) * gap_x
    assert base_h <= 28 and base_w <= 30
    top = int(rng.integers(1, 30 - base_h))
    bottom = int(rng.integers(1, 31 - base_h - top))
    left = int(rng.integers(0, 31 - base_w))
    right = int(rng.integers(0, 31 - base_w - left))
    grid = np.full((base_h + top + bottom, base_w + left + right), background, np.int8)
    slots = rng.choice(rows * cols, count, replace=False)
    for card, slot in zip(cards, slots):
        sr, sc = divmod(int(slot), cols)
        r = top + sr * (slot_h + gap_y) + int(rng.integers(slot_h - len(card) + 1))
        c = left + sc * (slot_w + gap_x) + int(rng.integers(slot_w - card.shape[1] + 1))
        grid[r:r + len(card), c:c + card.shape[1]] = card
    return grid


def _construct(rng, roles, m, n, count, witness):
    background, frame, a, b = roles
    signals = (a, b)
    tile, tile_majority = _tile(rng, n, signals)
    tile_card = np.pad(tile, 1, constant_values=frame)
    cards, projected = [tile_card], []
    turns = list(map(int, rng.integers(4, size=count)))
    if witness:
        turns[int(rng.integers(count))] = int(rng.choice([1, 3]))
    for i in range(count):
        board, labels = _source(rng, m, background, signals, witness and i == 0)
        projected.append(np.rot90(labels, -turns[i]))
        cards.append(_card(board, turns[i], background, frame, int(rng.choice(signals)),
                           int(rng.integers(m - 2))))
    # Forward labels come from component construction records, not the decoder.
    stack = np.stack(projected)
    av, bv = np.sum(stack == a, axis=0), np.sum(stack == b, axis=0)
    out = np.full((m * n, m * n), background, np.int8)
    for r in range(m):
        for c in range(m):
            if av[r, c] == bv[r, c]:
                continue
            winner = a if av[r, c] > bv[r, c] else b
            block = np.where(tile == tile_majority, a + b - winner, winner)
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = block
    return {"input": _pack(rng, cards, background), "output": out}


def _features(info):
    verdict, votes, population = _ballots(info)
    changed = {}
    for name in ("raw_pixels", "whole_sheet", "area_weighted", "no_rotation"):
        changed[name] = not np.array_equal(verdict, _ballots(info, name)[0])
    a, b = info["signals"]
    tied = (population > 0) & (votes[0] == votes[1])
    contested = (np.minimum(votes[0], votes[1]) > 0) & (votes[0] != votes[1])
    return dict(changed=changed, both_winners=bool(np.any(verdict == a) and np.any(verdict == b)),
                tied=int(tied.sum()), contested=int(contested.sum()),
                empty=int((population == 0).sum()),
                mixed_sheet=any(len(set(rec["majorities"])) > 1 for rec in info["sheets"]))


def _informative(info, require_area=False):
    f = _features(info)
    names = ("raw_pixels", "whole_sheet", "no_rotation") + (("area_weighted",) if require_area else ())
    return (f["both_winners"] and f["tied"] > 0 and f["contested"] > 0
            and f["mixed_sheet"] and all(f["changed"][name] for name in names))


def _arrow_conventions(infos, pairs):
    return [(sign, offset) for sign in (1, -1) for offset in range(4)
            if all(np.array_equal(_expand(info, _ballots(info, convention=(sign, offset))[0]),
                                  np.asarray(pair["output"]))
                   for info, pair in zip(infos, pairs))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One invariant rule; held-out sheet count, board scale, or tile scale."""
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))
    stable_palette = bool(rng.integers(2))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = split == "train" and i < 2
            roles = palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            for attempt in range(384):
                k = 3 if witness else int(rng.integers(2, 6))
                m = int(rng.integers(4 if witness else 3, 8))
                n = int(rng.integers(2, min(5, 30 // m) + 1))
                if shift == 0:
                    k = int(rng.integers(4, 6)) if split == "test" else (3 if witness else int(rng.integers(2, 4)))
                elif shift == 1:
                    m = int(rng.integers(6, 8)) if split == "test" else int(rng.integers(4 if witness else 3, 6))
                    n = int(rng.integers(2, min(5, 30 // m) + 1))
                else:
                    n = int(rng.integers(4, 6)) if split == "test" else int(rng.integers(2, 4))
                    m = min(m, 30 // n)
                pair = _construct(rng, roles, m, n, k, witness)
                if _key(pair["input"]) in seen:
                    continue
                info = _decode(pair["input"])
                if witness and not _informative(info, require_area=i == 1):
                    continue
                if witness and i == 1:
                    first = puzzle["train"][0]
                    if _arrow_conventions([_decode(first["input"]), info], [first, pair]) != [(1, 0)]:
                        continue
                # A nonempty output is necessary; sparse/tied-only boards are not useful examples.
                if np.all(pair["output"] == roles[0]):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded informative sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected = _inverse_prediction(info)
            assert np.array_equal(out, expected), "incorrect spatial vote or negative tile"
            assert np.array_equal(solve(inp), out)
            assert np.any(out != info["background"])
            if split == "train":
                train_info.append(info)
    assert any(_informative(info) for info in train_info), "missing two-level voting evidence"
    assert any(_features(info)["changed"]["area_weighted"] for info in train_info)
    # Compare a finite, stated hypothesis class for the arrow convention.
    assert _arrow_conventions(train_info[:2], puzzle["train"][:2]) == [(1, 0)], \
        "first two demonstrations leave turn semantics ambiguous"
    for behavior in ("rotate_tile", "no_exchange"):
        assert any(not np.array_equal(_expand(info, _ballots(info)[0], behavior),
                                     np.asarray(pair["output"]))
                   for info, pair in zip(train_info, puzzle["train"]))


def self_test():
    """42 puzzles streamed one at a time; fingerprints and counters only retained."""
    shapes, structures, palettes, fingerprints = set(), set(), set(), set()
    sizes, boards, sheet_counts, turns, region_counts = set(), set(), set(), set(), set()
    totals = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                f = _features(info)
                sizes.add(len(info["tile"]))
                boards.add(info["m"])
                sheet_counts.add(len(info["sheets"]))
                turns.update(rec["turn"] for rec in info["sheets"])
                region_counts.update(len(rec["regions"]) for rec in info["sheets"])
                palettes.add((info["background"], info["frame"], *info["signals"]))
                totals["pairs"] += 1
                totals["tied_positions"] += f["tied"]
                totals["contested_decisions"] += f["contested"]
                totals["mixed_sheets"] += f["mixed_sheet"]
                for name, changed in f["changed"].items():
                    totals[name + "_counterexamples"] += changed
                for role in ("input", "output"):
                    grid = pair[role]
                    shapes.add((role, *grid.shape))
                    roles = {}
                    normalized = bytes(roles.setdefault(int(v), len(roles)) for v in grid.flat)
                    structures.add((grid.shape, hashlib.sha256(normalized).hexdigest()))
                    digest.update(str(grid.shape).encode())
                    digest.update(grid.tobytes())
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42 and len(structures) >= 200 and len(shapes) >= 30
    assert sizes == {2, 3, 4, 5} and boards == {3, 4, 5, 6, 7}
    assert sheet_counts == {2, 3, 4, 5} and turns == {0, 1, 2, 3}
    assert max(region_counts) >= 3 and len(palettes) >= 30
    for key in ("raw_pixels", "whole_sheet", "area_weighted", "no_rotation"):
        assert totals[key + "_counterexamples"] >= 42
    p, q = generate_puzzle(913, 2, 1), generate_puzzle(913, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split])
               for k in ("input", "output"))
    generate_puzzle(914, 10, 4)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    inp, out = p["train"][0]["input"], p["train"][0]["output"]
    info = _decode(inp)
    # Color identities cannot affect roles, direction, or ballot outcomes.
    permutation = np.random.default_rng(7).permutation(10)
    assert np.array_equal(solve(permutation[inp]), permutation[out])
    # Sources commute: swapping two same-size cards cannot change the answer.
    changed = inp.copy()
    r, c, h, w = info["sheets"][0]["box"]
    y, x, _, _ = info["sheets"][1]["box"]
    changed[r:r + h, c:c + w] = inp[y:y + h, x:x + w]
    changed[y:y + h, x:x + w] = inp[r:r + h, c:c + w]
    assert np.array_equal(solve(changed), out)
    # Rearrange a source's votes, preserving each connected patch's counts.
    changed = inp.copy()
    rec = info["sheets"][0]
    r, c, _, _ = rec["box"]
    for cells in rec["regions"]:
        values = [int(rec["board"][p]) for p in cells][::-1]
        for (dr, dc), value in zip(cells, values):
            changed[r + 1 + dr, c + 1 + dc] = value
    assert np.array_equal(solve(changed), out)
    # Rotate a sheet CW and compensate with one less CW control turn.
    changed = inp.copy()
    m = info["m"]
    changed[r + 1:r + 1 + m, c + 1:c + 1 + m] = np.rot90(rec["board"], -1)
    kr, kc = rec["key_origin"]
    marker = int(changed[kr + 1, kc + 1])
    changed[kr:kr + 3, kc:kc + 3] = info["background"]
    changed[kr + 1, kc + 1] = marker
    dr, dc = DIRS[(rec["turn"] - 1) % 4]
    changed[kr + 1 + dr, kc + 1 + dc] = marker
    assert np.array_equal(solve(changed), out)
    damaged = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return dict(seeds=42, **dict(totals), distinct_puzzles=len(fingerprints),
                color_normalized_structures=len(structures), grid_shapes=len(shapes),
                palettes=len(palettes), tile_sizes=sorted(sizes), board_sizes=sorted(boards),
                sheet_counts=sorted(sheet_counts), turns=sorted(turns),
                region_counts=sorted(region_counts), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
