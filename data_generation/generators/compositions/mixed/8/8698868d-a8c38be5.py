"""Address nine carriers, assemble them, then transplant cavity-count prototypes.

Only standard-library modules and NumPy are used. No parent module, generation
metadata, seed, or example output is consulted by the input-only decoder.
"""
import hashlib
import json

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SLOTS = tuple((r, c) for r in range(3) for c in range(3))
_MASKS = {
    (0, 0): ((0, 0), (0, 1), (1, 0)),
    (0, 1): ((0, 0), (0, 1), (0, 2), (1, 1)),
    (0, 2): ((0, 1), (0, 2), (1, 2)),
    (1, 0): ((0, 0), (1, 0), (2, 0), (1, 1)),
    (1, 1): (),
    (1, 2): ((0, 2), (1, 2), (2, 2), (1, 1)),
    (2, 0): ((1, 0), (2, 0), (2, 1)),
    (2, 1): ((1, 1), (2, 0), (2, 1), (2, 2)),
    (2, 2): ((1, 2), (2, 1), (2, 2)),
}
_CAPACITY = {5: 1, 6: 2, 7: 5}


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _address(size, slot):
    # Embed the parent's 3x3 address lattice in a larger carrier. For even
    # sizes the middle anchor is the upper/left of the two central cells.
    anchors = (0, (size - 1) // 2, size - 1)
    return frozenset((anchors[r], anchors[c]) for r, c in _MASKS[slot])


def _variants(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in _variants(mask)[1:])


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _interpret(a, neutral):
    groups = _components(a != neutral)
    assert 11 <= len(groups) <= 15
    boxes = [_box(g) for g in groups]
    sides = {h for r, c, h, w in boxes if h == w}
    assert len(sides) == 2 and all(h == w for r, c, h, w in boxes)
    m, t = sorted(sides)
    assert t in _CAPACITY and m == t - 2
    tile_boxes = sorted(b for b in boxes if b[2] == t)
    motif_boxes = sorted(b for b in boxes if b[2] == m)
    assert len(tile_boxes) == 9 and 2 <= len(motif_boxes) <= _CAPACITY[t] + 1
    for i, (r, c, h, w) in enumerate(boxes):
        for rr, cc, hh, ww in boxes[i + 1:]:
            assert r + h < rr or rr + hh < r or c + w < cc or cc + ww < c
    carriers = []
    for r, c, _, _ in tile_boxes:
        tile = a[r:r + t, c:c + t]
        freq = np.bincount(tile[tile != neutral], minlength=10)
        carrier = int(np.argmax(freq))
        assert freq[carrier] > t * t // 2
        carriers.append(carrier)
    assert len(set(carriers)) == 1
    carrier = carriers[0]
    assert carrier != neutral
    addresses = {_address(t, slot): slot for slot in _SLOTS}
    tiles, accents, clue_masks = {}, {}, {}
    for r, c, _, _ in tile_boxes:
        tile = a[r:r + t, c:c + t]
        clues = tile == neutral
        assert not np.any(_perimeter(clues))
        assert not np.any(clues[:-1] & clues[1:])
        assert not np.any(clues[:, :-1] & clues[:, 1:])
        other = (tile != carrier) & ~clues
        mask = frozenset(map(tuple, np.argwhere(other)))
        assert mask in addresses
        slot = addresses[mask]
        assert slot not in tiles
        colors = np.unique(tile[other])
        assert len(colors) == int(slot != (1, 1))
        tiles[slot] = (r, c, int(clues.sum()))
        accents[slot] = None if not len(colors) else int(colors[0])
        clue_masks[slot] = clues[1:-1, 1:-1].copy()
    assert set(tiles) == set(_SLOTS)
    motifs = {}
    for r, c, _, _ in motif_boxes:
        motif = a[r:r + m, c:c + m]
        colors = np.unique(motif[motif != neutral])
        assert len(colors) == 1 and int(colors[0]) not in (neutral, carrier)
        assert np.all(_perimeter(motif) == colors[0])
        k = len(_components(motif == neutral))
        assert k not in motifs and 0 <= k <= _CAPACITY[t]
        motifs[k] = motif.copy()
    assert {v[2] for v in tiles.values()} == set(motifs)
    return {"neutral": neutral, "carrier": carrier, "size": t, "tiles": tiles,
            "accents": accents, "motifs": motifs, "clues": clue_masks,
            "tile_boxes": tile_boxes, "motif_boxes": motif_boxes}


def _decode(input_grid):
    a = _grid(input_grid)
    interpretations = []
    for neutral in map(int, np.unique(a)):
        try:
            interpretations.append(_interpret(a, neutral))
        except (AssertionError, ValueError, IndexError):
            pass
    if len(interpretations) != 1:
        raise ValueError("input must have one neutral/carrier/address/library interpretation")
    return interpretations[0]


def _assembled_input(a, info):
    t = info["size"]
    assembled = np.empty((3 * t, 3 * t), dtype=np.int8)
    for (sr, sc), (r, c, k) in info["tiles"].items():
        assembled[sr * t:(sr + 1) * t, sc * t:(sc + 1) * t] = a[r:r + t, c:c + t]
    return assembled


def _transplant(assembled, neutral, carrier, motifs):
    """Read clue counts after assembly; then consume all address/clue ink."""
    t = assembled.shape[0] // 3
    result = np.full_like(assembled, carrier)
    for sr, sc in _SLOTS:
        tile = assembled[sr * t:(sr + 1) * t, sc * t:(sc + 1) * t]
        k = int(np.count_nonzero(tile == neutral))
        motif = motifs[k]
        result[sr * t + 1:(sr + 1) * t - 1, sc * t + 1:(sc + 1) * t - 1] = np.where(
            motif == neutral, carrier, motif)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _decode(a)
    assembled = _assembled_input(a, info)
    return _transplant(assembled, info["neutral"], info["carrier"], info["motifs"])


def _holes(rng, m, k, witness=False):
    mask = np.zeros((m, m), dtype=bool)
    if not k:
        return mask
    if witness and k == 1:
        mask[1:-1, 1:-1] = True
        return mask
    if witness and k == 2:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
        return _pick(rng, _variants(mask)).copy()
    if k == 1 and rng.random() < .30:
        mask[1:-1, 1:-1] = True
        return mask
    # A checkerboard subset is a capacity-guaranteed starting point; try a
    # general independent set first, then grow arbitrary connected cavities.
    cells = [(r, c) for r in range(1, m - 1) for c in range(1, m - 1)]
    found = False
    for _ in range(12):
        mask[:] = False
        for index in rng.permutation(len(cells)):
            r, c = cells[int(index)]
            if any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
            if int(mask.sum()) == k:
                found = True
                break
        if found:
            break
    if not found:
        mask[:] = False
        fallback = [(r, c) for r, c in cells if (r + c) % 2 == 0]
        for index in rng.permutation(len(fallback))[:k]:
            mask[fallback[int(index)]] = True
    target = int(rng.integers(k, (m - 2) ** 2 + 1))
    for _ in range((m - 2) ** 2):
        if int(mask.sum()) >= target:
            break
        choices = []
        for r, c in cells:
            if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                continue
            trial = mask.copy()
            trial[r, c] = True
            if len(_components(trial)) == k and len(_components(~trial)) == 1:
                choices.append((r, c))
        if not choices:
            break
        mask[_pick(rng, choices)] = True
    assert len(_components(mask)) == k and len(_components(~mask)) == 1
    return mask


def _clues(rng, t, count, reserved):
    cells = [(r, c) for r in range(1, t - 1) for c in range(1, t - 1)
             if (r, c) not in reserved]
    for _ in range(24):
        mask = np.zeros((t, t), dtype=bool)
        if count == 0:
            return mask
        for index in rng.permutation(len(cells)):
            r, c = cells[int(index)]
            if any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
            if int(mask.sum()) == count:
                return mask
    # Sparse corners-and-center lattice, excluding any inward address cell.
    mask = np.zeros((t, t), dtype=bool)
    fallback = [(r, c) for r, c in cells if r % 2 == c % 2 == 1]
    assert len(fallback) >= count
    for index in rng.permutation(len(fallback))[:count]:
        mask[fallback[int(index)]] = True
    return mask


def _free_origins(boxes, size):
    rr, cc = np.indices((31 - size, 31 - size))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, h, w in boxes:
        free &= (rr + size < r) | (r + h < rr) | (cc + size < c) | (c + w < cc)
    return np.argwhere(free)


def _scatter(rng, objects, neutral, t):
    m = t - 2
    boxes = [(r * (t + 1), c * (t + 1), t, t) for r, c in _SLOTS]
    edge = 3 * t + 3
    shelves = [(i * (m + 1), edge, m, m) for i in range(4)]
    shelves += [(edge, i * (m + 1), m, m) for i in range(5)]
    boxes += [shelves[int(i)] for i in rng.permutation(9)[:len(objects) - 9]]
    # Relocate each object independently from a guaranteed feasible packing.
    # Contents are never reflected or rotated by this layout operation.
    if rng.integers(2):
        boxes = [(c, r, w, h) for r, c, h, w in boxes]
    if rng.integers(2):
        boxes = [(30 - r - h, c, h, w) for r, c, h, w in boxes]
    if rng.integers(2):
        boxes = [(r, 30 - c - w, h, w) for r, c, h, w in boxes]
    for _ in range(2):
        for index in rng.permutation(len(boxes)):
            index = int(index)
            size = boxes[index][2]
            free = _free_origins(boxes[:index] + boxes[index + 1:], size)
            r, c = map(int, free[int(rng.integers(len(free)))])
            boxes[index] = (r, c, size, size)
    a = np.full((30, 30), neutral, dtype=np.int8)
    for obj, (r, c, h, w) in zip(objects, boxes):
        a[r:r + h, c:c + w] = obj
    r0, c0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    r1, c1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    r0, c0 = int(rng.integers(r0 + 1)), int(rng.integers(c0 + 1))
    r1, c1 = int(rng.integers(r1, 31)), int(rng.integers(c1, 31))
    return a[r0:r1, c0:c1].copy()


def _make_pair(rng, t, counts, roles, accent_number, witness=False, previous=None):
    neutral, carrier = roles
    palette = [c for c in range(10) if c not in roles]
    accent_pool = list(map(int, rng.choice(palette, accent_number, replace=False)))
    # Each requested accent identity appears; additional labels reuse it.
    accent_values = accent_pool + list(map(int, rng.choice(accent_pool, 8 - accent_number)))
    rng.shuffle(accent_values)
    labels = iter(accent_values)
    assignments = None
    for _ in range(96):
        candidates = []
        for i in range(9):
            allowed = [k for k in counts if previous is None or k != previous[i]]
            candidates.append(_pick(rng, allowed))
        if set(candidates) == set(counts):
            assignments = candidates
            break
    if assignments is None:
        # A bounded exhaustive assignment fallback (at most 6**9 candidates
        # in principle, but stop as soon as all required counts are covered).
        def search(i, values, missing):
            if i == 9:
                return values if not missing else None
            allowed = [k for k in counts if previous is None or k != previous[i]]
            allowed.sort(key=lambda k: k not in missing)
            for k in allowed:
                if len(missing - {k}) <= 8 - i:
                    result = search(i + 1, values + [k], missing - {k})
                    if result is not None:
                        return result
            return None
        assignments = search(0, [], set(counts))
        assert assignments is not None
    m = t - 2
    motifs = {}
    for k in counts:
        mask = _holes(rng, m, k, witness)
        motif = np.full((m, m), _pick(rng, palette), dtype=np.int8)
        motif[mask] = neutral
        motifs[k] = motif
    tiles = []
    expected = np.full((3 * t, 3 * t), carrier, dtype=np.int8)
    for slot, k in zip(_SLOTS, assignments):
        tile = np.full((t, t), carrier, dtype=np.int8)
        address = _address(t, slot)
        if address:
            accent = next(labels)
            for p in address:
                tile[p] = accent
        tile[_clues(rng, t, k, address)] = neutral
        tiles.append(tile)
        r, c = slot
        expected[r * t + 1:(r + 1) * t - 1, c * t + 1:(c + 1) * t - 1] = np.where(
            motifs[k] == neutral, carrier, motifs[k])
    # A count-changing shuffle guarantees that ignoring addresses changes the
    # final answer, even after all accent labels have been erased.
    for _ in range(32):
        order = list(map(int, rng.permutation(9)))
        if [assignments[i] for i in order] != assignments:
            break
    else:
        order = list(range(9))
        j = next(i for i in range(1, 9) if assignments[i] != assignments[0])
        order[0], order[j] = order[j], order[0]
    objects = [tiles[i] for i in order] + [motifs[k] for k in rng.permutation(counts)]
    inp = _scatter(rng, objects, neutral, t)
    # Physical raster order can change during relocation. Enforce the actual
    # observable shortcut failure rather than trusting the initial permutation.
    info = _decode(inp)
    if np.array_equal(_raster_prediction(inp, info), expected):
        boxes = info["tile_boxes"]
        i, j = next((i, j) for i in range(9) for j in range(i + 1, 9)
                    if int((inp[boxes[i][0]:boxes[i][0] + t, boxes[i][1]:boxes[i][1] + t] == neutral).sum())
                    != int((inp[boxes[j][0]:boxes[j][0] + t, boxes[j][1]:boxes[j][1] + t] == neutral).sum()))
        r, c = boxes[i][:2]
        rr, cc = boxes[j][:2]
        first = inp[r:r + t, c:c + t].copy()
        inp[r:r + t, c:c + t] = inp[rr:rr + t, cc:cc + t]
        inp[rr:rr + t, cc:cc + t] = first
    return {"input": inp, "output": expected}, assignments


def _raster_prediction(a, info):
    t = info["size"]
    assembled = np.empty((3 * t, 3 * t), dtype=np.int8)
    for (sr, sc), (r, c, _, _) in zip(_SLOTS, info["tile_boxes"]):
        assembled[sr * t:(sr + 1) * t, sc * t:(sc + 1) * t] = a[r:r + t, c:c + t]
    return _transplant(assembled, info["neutral"], info["carrier"], info["motifs"])


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(info):
    neutral = info["neutral"]
    masks = {k: motif == neutral for k, motif in info["motifs"].items()}
    areas = {k: int(mask.sum()) for k, mask in masks.items()}
    return {
        "area_inversion": any(a < b and areas[a] > areas[b] for a in areas for b in areas),
        "diagonal": any(len(_components(mask, True)) < k for k, mask in masks.items()),
        "asymmetric": any(_asymmetric(mask) for mask in masks.values()),
        "nonpositional": any(k > 0 and not np.array_equal(info["clues"][slot], masks[k])
                             for slot, (r, c, k) in info["tiles"].items()),
        "reuse": len(set(v for v in info["accents"].values() if v is not None)) < 8,
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _counts(nt, ne)
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _decode(a)
            assembled = _assembled_input(a, info)
            expected = _transplant(assembled, info["neutral"], info["carrier"], info["motifs"])
            assert np.array_equal(b, expected), "wrong address, motif, substitution, orientation, or crop"
            assert not np.any(b == info["neutral"])
            t = info["size"]
            for sr, sc in _SLOTS:
                tile = b[sr * t:(sr + 1) * t, sc * t:(sc + 1) * t]
                assert np.all(_perimeter(tile) == info["carrier"])
            assert not np.array_equal(assembled, expected), "omitting motif transplant has no effect"
            assert not np.array_equal(_raster_prediction(a, info), expected), "omitting address has no effect"
            infos[split].append(info)
    feats = [_features(info) for info in infos["train"]]
    for name in ("area_inversion", "diagonal", "asymmetric", "nonpositional", "reuse"):
        assert any(f[name] for f in feats), "missing training witness: " + name
    # Every destination is demonstrated with more than one clue count. This
    # refutes a fixed slot-to-prototype map independently of color identity.
    for slot in _SLOTS:
        assert len({info["tiles"][slot][2] for info in infos["train"]}) >= 2
    assert all(5 not in info["motifs"] for info in infos["train"])
    assert all(5 in info["motifs"] for info in infos["test"])


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable_roles = bool(rng.integers(2))
    common_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    first_counts = None
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = split == "train" and i == 0
            if witness:
                t = 7
                counts = [0, 1, 2] + [k for k in (3, 4) if rng.random() < .5]
            elif split == "train":
                t = int(rng.choice((5, 6, 7), p=(.20, .30, .50)))
                allowed = list(range(min(4, _CAPACITY[t]) + 1))
                counts = sorted(map(int, rng.choice(allowed, int(rng.integers(2, len(allowed) + 1)), replace=False)))
            else:
                t = 7
                counts = sorted([5] + list(map(int, rng.choice(5, int(rng.integers(3, 6)), replace=False))))
            if witness:
                accent_number = int(rng.integers(2, 9))
            elif split == "train" and i == 1:
                accent_number = 1
            else:
                accent_number = int(rng.integers(1, 9 if split == "train" else 4))
            for attempt in range(24):
                roles = common_roles if stable_roles else tuple(map(int, rng.choice(10, 2, replace=False)))
                if not stable_roles and split == "train" and i == 1:
                    roles = _pick(rng, [p for p in ((a, b) for a in range(10) for b in range(10) if a != b)
                                        if p != first_roles])
                pair, assignments = _make_pair(rng, t, counts, roles, accent_number, witness,
                                               first_counts if split == "train" and i == 1 else None)
                key = _key(pair["input"])
                if key not in used:
                    puzzle[split].append(pair)
                    used.add(key)
                    if witness:
                        first_counts, first_roles = assignments, roles
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain hashes/metrics, never full run collections."""
    shapes, sizes, count_sets, layouts, masks, roles, hashes = (set() for _ in range(7))
    totals = {"puzzles": 42, "pairs": 0, "area_witnesses": 0, "diagonal_witnesses": 0,
              "asymmetric_witnesses": 0, "address_ablation_failures": 0,
              "motif_ablation_failures": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                before = a.copy()
                assert np.array_equal(solve(a), b) and np.array_equal(a, before)
                info = _decode(a)
                feature = _features(info)
                shapes.add(a.shape)
                sizes.add(info["size"])
                roles.add((info["neutral"], info["carrier"]))
                count_sets.add(tuple(sorted(info["motifs"])))
                layouts.add(tuple(info["tiles"][slot][:2] for slot in _SLOTS))
                for motif in info["motifs"].values():
                    mask = motif == info["neutral"]
                    masks.add((mask.shape, min(v.tobytes() for v in _variants(mask))))
                totals["pairs"] += 1
                totals["area_witnesses"] += feature["area_inversion"]
                totals["diagonal_witnesses"] += feature["diagonal"]
                totals["asymmetric_witnesses"] += feature["asymmetric"]
                totals["address_ablation_failures"] += not np.array_equal(_raster_prediction(a, info), b)
                totals["motif_ablation_failures"] += not np.array_equal(_assembled_input(a, info), b)
    assert len(hashes) == 42 and sizes == {5, 6, 7}
    assert len(shapes) >= 10 and len(layouts) >= 180 and len(roles) >= 25
    assert len(count_sets) >= 12 and len(masks) >= 25
    assert totals["area_witnesses"] >= 42 and totals["diagonal_witnesses"] >= 42
    assert totals["asymmetric_witnesses"] >= 42
    assert totals["address_ablation_failures"] == totals["motif_ablation_failures"] == totals["pairs"]
    # Exercise all legal explicit counts, including the maximum 14-pair puzzle.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _fingerprint(generate_puzzle(314159, 2, 1)) == _fingerprint(generate_puzzle(314159, 2, 1))
    # Reject a semantically corrupted label.
    broken = generate_puzzle(12, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    totals.update({"input_shapes": len(shapes), "tile_sizes": sorted(sizes),
                   "count_sets": len(count_sets), "layouts": len(layouts),
                   "cavity_shapes_up_to_D4": len(masks), "role_pairs": len(roles),
                   "distinct_puzzle_hashes": len(hashes)})
    return totals


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
