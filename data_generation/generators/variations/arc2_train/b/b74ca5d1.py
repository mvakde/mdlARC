"""Corner-keyed unions of registered five-by-five stencils."""

import hashlib
import numpy as np

SIZE = 30
CORNER_CELLS = ((0, 0), (0, 29), (29, 0), (29, 29))
WINDOWS = ((0, 0), (0, 25), (25, 0), (25, 25))
LOCAL_CORNERS = ((0, 0), (0, 4), (4, 0), (4, 4))
SOURCE_HASHES = frozenset((
    "a2ffa8bf95ecc1351839894d397dc65365d162db1166e29d06aa0680dd812536",
    "8f3d8d3e1622539a59dad794e54a06f10a1368c5ecf56f7a1f18f47d8f9b074e",
    "ae6727d6df350609632da023d8a70616d4498733c671054618777e1b1d1c65cd",
    "3e6abfe98bd297f85601813e5f8b087680db9ee4d5935d0c04591aa25c0c6cd8",
    "63959a401126bf4cd2e17f083f4dc0e94f9ba89a6714105554edd9396e92c234",
))


def _counts(seed, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lo = max(2, num_test + 1)
        num_train = lo + (seed // 4 + 3 * (seed % 7)) % (11 - lo)
    elif num_train is None:
        lo = max(2, num_test + 1)
        num_train = lo + (5 * seed + 17) % (11 - lo)
    else:
        num_test = 1 + (7 * seed + 11) % min(4, num_train - 1)
    return int(num_train), int(num_test)


def _tight(mask):
    pts = np.argwhere(mask)
    return (len(pts) > 0 and pts[:, 0].min() == 0 and pts[:, 0].max() == 4
            and pts[:, 1].min() == 0 and pts[:, 1].max() == 4)


def _shape(rng, corner, multi):
    rr, cc = np.indices((5, 5))
    mode = int(rng.integers(0, 12))
    if mode == 0:
        mask = rr == cc
    elif mode == 1:
        mask = rr + cc == 4
    elif mode == 2:
        mask = (rr == cc) | (rr + cc == 4)
    elif mode == 3:
        mask = (rr == 2) | (cc == 2)
    elif mode == 4:
        mask = (rr % 2 == 0) | (cc == int(rng.integers(0, 5)))
    elif mode == 5:
        mask = (cc % 2 == 0) | (rr == int(rng.integers(0, 5)))
    elif mode == 6:
        mask = ((rr == 0) | (rr == 4) | (cc == 0) | (cc == 4))
        mask &= rng.random((5, 5)) < 0.84
    elif mode == 7:
        mask = ((rr == 0) | (rr == 4) | (cc == 0) | (cc == 4)
                | ((rr + cc + int(rng.integers(0, 2))) % 2 == 0))
    elif mode == 8:
        mask = (rr + cc + int(rng.integers(0, 2))) % 2 == 0
    elif mode == 9:
        mask = rng.random((5, 5)) < float(rng.uniform(0.28, 0.49))
    elif mode == 10:
        mask = rng.random((5, 5)) < float(rng.uniform(0.52, 0.83))
    else:
        mask = np.ones((5, 5), dtype=bool)
        if rng.random() < 0.8:
            mask &= rng.random((5, 5)) < 0.90
    mask = np.asarray(mask, dtype=bool)
    mask[LOCAL_CORNERS[corner]] = True
    for edge in (mask[0, :], mask[4, :], mask[:, 0], mask[:, 4]):
        if not edge.any():
            edge[int(rng.integers(0, 5))] = True
    if mask.sum() < 5:
        free = np.argwhere(~mask)
        rng.shuffle(free)
        for r, c in free[:5 - int(mask.sum())]:
            mask[r, c] = True
    if multi and mask.sum() < 10:
        free = np.argwhere(~mask)
        rng.shuffle(free)
        for r, c in free[:10 - int(mask.sum())]:
            mask[r, c] = True
    if not multi and mask.all():
        mask[2, 2] = False
    return mask


def _split(union, count, overlap, rng):
    cells = [tuple(int(v) for v in p) for p in np.argwhere(union)]
    sides = (lambda p: p[0] == 0, lambda p: p[0] == 4,
             lambda p: p[1] == 0, lambda p: p[1] == 4)
    for _ in range(32):
        masks = np.zeros((count, 5, 5), dtype=bool)
        private = rng.choice(len(cells), size=count, replace=False)
        owners = {cells[int(j)]: i for i, j in enumerate(private)}
        for p, owner in owners.items():
            masks[owner][p] = True
        good = True
        for i in range(count):
            for side in sides:
                if any(side(p) and masks[i][p] for p in cells):
                    continue
                available = [p for p in cells if side(p) and owners.get(p, i) == i
                             and (overlap > 0 or not masks[:, p[0], p[1]].any())]
                if not available:
                    good = False
                    break
                masks[i][available[int(rng.integers(0, len(available)))]] = True
            if not good:
                break
        if not good:
            continue
        for p in cells:
            if p in owners:
                continue
            if not masks[:, p[0], p[1]].any():
                masks[int(rng.integers(0, count))][p] = True
            if overlap:
                for i in rng.permutation(count):
                    if rng.random() < overlap:
                        masks[int(i)][p] = True
        if (all(_tight(m) and m.sum() >= 5 and not np.array_equal(m, union) for m in masks)
                and (overlap == 0 or np.any(masks.sum(axis=0) > 1))
                and np.array_equal(masks.any(axis=0), union)):
            return [m.copy() for m in masks]
    return None


def _family_masks(rng, corner, count, forced=None):
    if count == 1:
        return [_shape(rng, corner, False)]
    overlap = ((0.0, 0.18, 0.42, 0.72)[int(rng.integers(0, 4))]
               if forced is None else forced)
    for _ in range(24):
        result = _split(_shape(rng, corner, True), count, overlap, rng)
        if result is not None:
            return result
    for _ in range(64):
        result = _split(np.ones((5, 5), dtype=bool), count, overlap, rng)
        if result is not None:
            return result
    raise AssertionError("could not split union")


def _positions(rng, count):
    candidates = [(r, c) for r in range(2, 24) for c in range(2, 24)
                  if all(r + 4 < wr - 1 or r > wr + 5
                         or c + 4 < wc - 1 or c > wc + 5
                         for wr, wc in WINDOWS)]
    for _ in range(48):
        chosen = []
        for j in rng.permutation(len(candidates)):
            point = candidates[int(j)]
            if all(abs(point[0] - q[0]) >= 8
                   or abs(point[1] - q[1]) >= 8 for q in chosen):
                chosen.append(point)
                if len(chosen) == count:
                    return chosen
    raise AssertionError("could not place separated tiles")


def _profiles(rng, regime, index, order):
    if regime == "source_train" and index < 2:
        if index == 0:
            active = [int(v) for v in order[:3]]
            counts = [2, 1, int(rng.integers(1, 3))]
            forced, forced_count = 0.0, 2
        else:
            active = [int(order[3])] + [int(v) for v in rng.choice(order[:3], 2, replace=False)]
            counts = [3, 1, int(rng.integers(1, 3))]
            forced, forced_count = 0.55, 3
        rng.shuffle(active)
        rng.shuffle(counts)
        force_index = counts.index(forced_count)
    else:
        active_count = (2 + int(rng.integers(0, 2)) if regime == "source_train"
                        else int(rng.choice((1, 2, 2, 2, 3, 3, 3, 4))))
        active = [int(v) for v in rng.choice(4, active_count, replace=False)]
        counts = [int(rng.choice((1, 1, 2, 2, 3))) for _ in active]
        while sum(counts) > 8:
            eligible = [i for i, value in enumerate(counts) if value > 1]
            j = eligible[int(rng.integers(0, len(eligible)))]
            counts[j] -= 1
        forced, force_index = None, -1
    return [(corner, n, forced if i == force_index else None)
            for i, (corner, n) in enumerate(zip(active, counts))]


def _make_pair(rng, regime, index, order, palette):
    families = []
    for corner, count, forced in _profiles(rng, regime, index, order):
        masks = _family_masks(rng, corner, count, forced)
        tags = []
        for mask in masks:
            pts = np.argwhere(mask)
            r, c = pts[int(rng.integers(0, len(pts)))]
            tags.append((int(r), int(c)))
        families.append((corner, masks, tags))
    total = sum(len(masks) for _, masks, _ in families)
    positions = _positions(rng, total)
    colors = rng.permutation(palette)
    bg = int(colors[0])
    keys = [int(v) for v in colors[1:1 + len(families)]]
    pool = [int(v) for v in colors[1 + len(families):]]
    if rng.random() < 0.38:
        carriers = [pool[int(rng.integers(0, min(len(pool), 4)))] for _ in range(total)]
    else:
        carriers = [pool[i % len(pool)] for i in range(total)]
        rng.shuffle(carriers)
    inp = np.full((SIZE, SIZE), bg, dtype=np.int8)
    out = inp.copy()
    cursor = 0
    for (corner, masks, tags), key in zip(families, keys):
        inp[CORNER_CELLS[corner]] = key
        union = np.any(np.stack(masks), axis=0)
        wr, wc = WINDOWS[corner]
        out[wr:wr + 5, wc:wc + 5][union] = key
        for mask, tag in zip(masks, tags):
            r, c = positions[cursor]
            carrier = carriers[cursor]
            inp[r:r + 5, c:c + 5][mask] = carrier
            inp[r + tag[0], c + tag[1]] = key
            out[r:r + 5, c:c + 5][mask] = key
            out[r + tag[0], c + tag[1]] = carrier
            cursor += 1
    return {"input": inp, "output": out}


def _parse_pair(pair):
    inp, out = pair["input"], pair["output"]
    bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
    active = [(i, int(inp[p])) for i, p in enumerate(CORNER_CELLS) if int(inp[p]) != bg]
    assert 1 <= len(active) <= 4
    keys = {key for _, key in active}
    assert len(keys) == len(active)
    tags = [tuple(int(v) for v in p) for p in np.argwhere(np.isin(inp, list(keys)))
            if tuple(p) not in CORNER_CELLS]
    assert len(tags) >= len(active)
    tiles = []
    used = np.zeros(inp.shape, dtype=bool)
    for tr, tc in tags:
        choices = []
        for r in range(max(1, tr - 4), min(tr, 25) + 1):
            for c in range(max(1, tc - 4), min(tc, 25) + 1):
                tile = inp[r:r + 5, c:c + 5]
                mask = tile != bg
                if not _tight(mask):
                    continue
                local_tags = np.argwhere(np.isin(tile, list(keys)))
                if len(local_tags) != 1 or (r + int(local_tags[0, 0]), c + int(local_tags[0, 1])) != (tr, tc):
                    continue
                carriers = set(int(v) for v in tile[mask]) - keys
                if len(carriers) != 1:
                    continue
                r0, c0 = max(0, r - 1), max(0, c - 1)
                border = inp[r0:min(30, r + 6), c0:min(30, c + 6)].copy()
                border[r - r0:r - r0 + 5, c - c0:c - c0 + 5] = bg
                if np.any(border != bg):
                    continue
                choices.append((r, c, mask, int(tile[tuple(local_tags[0])]),
                                carriers.pop(), tuple(int(v) for v in local_tags[0])))
        assert len(choices) == 1, f"tag {(tr, tc)} has {len(choices)} windows: {[(x[0], x[1]) for x in choices]}"
        r, c, mask, key, carrier, tag = choices[0]
        assert not used[r:r + 5, c:c + 5].any()
        used[r:r + 5, c:c + 5] = True
        tiles.append((r, c, mask, key, carrier, tag))
    possible = used.copy()
    for p in CORNER_CELLS:
        possible[p] = True
    assert np.all((inp != bg) <= possible)
    expected = np.full(inp.shape, bg, dtype=np.int8)
    sizes, unions, overlaps = [], [], []
    shared = full = 0
    for corner, key in active:
        family = [t for t in tiles if t[3] == key]
        assert family
        masks = np.stack([t[2] for t in family])
        union = masks.any(axis=0)
        assert union[LOCAL_CORNERS[corner]]
        if len(family) > 1:
            assert all(not np.array_equal(m, union) for m in masks)
        multiplicity = masks.sum(axis=0)
        shared += sum(int(multiplicity[t[5]] > 1) for t in family)
        sizes.append(len(family))
        unions.append(int(union.sum()))
        overlaps.append(int((multiplicity > 1).sum()))
        full += int(union.all())
        wr, wc = WINDOWS[corner]
        expected[wr:wr + 5, wc:wc + 5][union] = key
        for r, c, mask, _, carrier, tag in family:
            expected[r:r + 5, c:c + 5][mask] = key
            expected[r + tag[0], c + tag[1]] = carrier
    assert np.array_equal(expected, out)
    return {"corners": tuple(i for i, _ in active), "family_sizes": sizes,
            "union_sizes": unions, "overlap_sizes": overlaps, "shared_tags": shared,
            "full": full, "positions": tuple((t[0], t[1]) for t in tiles),
            "carriers": tuple(t[4] for t in tiles)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu" and grid.shape == (30, 30)
                assert np.all((0 <= grid) & (grid <= 9))
            assert not np.array_equal(pair["input"], pair["output"])
            identity = (pair["input"].shape, pair["input"].tobytes())
            assert identity not in seen
            seen.add(identity)
            digest = hashlib.sha256(pair["input"].astype(np.int8).tobytes()).hexdigest()
            assert digest not in SOURCE_HASHES
            analyses[split].append(_parse_pair(pair))
    train = analyses["train"]
    train_corners = set().union(*(set(a["corners"]) for a in train))
    assert train_corners == set(range(4))
    assert all(2 <= len(a["corners"]) <= 3 for a in train)
    assert any(2 in a["family_sizes"] for a in train)
    assert any(3 in a["family_sizes"] for a in train)
    assert any(0 in a["overlap_sizes"] for a in train)
    assert any(any(v > 0 for v in a["overlap_sizes"]) for a in train)
    assert all(set(a["corners"]) <= train_corners for a in analyses["test"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    nt, ne = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    order = rng.permutation(4)
    palette = rng.permutation(10)
    puzzle = {"train": [], "test": []}
    for split, count, regime in (("train", nt, "source_train"),
                                 ("test", ne, "target_test")):
        for i in range(count):
            for attempt in range(40):
                try:
                    pair = _make_pair(rng, regime, i, order, palette)
                    _parse_pair(pair)
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError(f"could not construct unambiguous pair {split} {i}")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert any(not np.array_equal(puzzles[0]["train"][0][part],
                                  puzzles[1]["train"][0][part])
               for part in ("input", "output"))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(123, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(123, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    a, b = generate_puzzle(123), generate_puzzle(123)
    assert all(np.array_equal(a[s][i][part], b[s][i][part])
               for s in ("train", "test") for i in range(len(a[s]))
               for part in ("input", "output"))
    by_split = {s: [_parse_pair(pair) for puzzle in puzzles for pair in puzzle[s]]
                for s in ("train", "test")}
    analyses = by_split["train"] + by_split["test"]
    positions = {p for a in analyses for p in a["positions"]}
    union_sizes = {v for a in analyses for v in a["union_sizes"]}
    family_sizes = {v for a in analyses for v in a["family_sizes"]}
    shapes = set()
    backgrounds = set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                backgrounds.add(int(np.bincount(inp.ravel(), minlength=10).argmax()))
                for corner, point in enumerate(CORNER_CELLS):
                    key = int(inp[point])
                    if key == int(np.bincount(inp.ravel(), minlength=10).argmax()):
                        continue
                    wr, wc = WINDOWS[corner]
                    shapes.add((out[wr:wr + 5, wc:wc + 5] == key).tobytes())
    assert len(positions) >= 35 and len(union_sizes) >= 12
    assert len(shapes) >= 100 and len(backgrounds) >= 8
    assert family_sizes >= {1, 2, 3}
    for split in ("train", "test"):
        group = by_split[split]
        assert any(a["shared_tags"] for a in group)
        assert any(a["full"] for a in group)
        assert any(len(set(a["carriers"])) < len(a["carriers"]) for a in group)
        assert any(n > 1 and u <= 15 for a in group
                   for n, u in zip(a["family_sizes"], a["union_sizes"]))
        assert any(n > 1 and overlap == 0 for a in group
                   for n, overlap in zip(a["family_sizes"], a["overlap_sizes"]))
        assert any(n > 1 and overlap > 0 for a in group
                   for n, overlap in zip(a["family_sizes"], a["overlap_sizes"]))
    assert {len(a["corners"]) for a in by_split["test"]} == {1, 2, 3, 4}
    assert any(v <= 15 for a in analyses for v in a["union_sizes"])
    assert any(v >= 22 for a in analyses for v in a["union_sizes"])
