"""Find a strict union relation between the plus-union masks of framed motifs.

Only the standard library and NumPy are used. No files, examples, seeds, or
construction metadata are consulted by solve(). Generation and checks do no I/O.
"""
from __future__ import annotations

import hashlib
import numpy as np

DIVIDER = 5
OPERAND = 8
RESULT = 3
PALETTE = (0, 1, 2, 4, 6, 7, 9)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 801437117, 4977645, 358033091, 281136089, 912297247,
              323301306, 516540721, 380157937, 927206100, 164630257,
              942900445, 168027897, 983569666, 227347951, 472258939,
              620594772, 553104938, 360284297, 670363955, 18864231,
              557022455, 936858950, 417245135, 398170845, 501663682,
              694818786, 364032893, 992889801, 820730412, 760703030,
              7548033, 293954355, 402764407, 494176530, 944141770,
              412434469, 910136845, 8605442, 336282125, 753772757, 953411491)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer 2D grid, size 1..30, colors 0..9")
    return grid


def _plus_centers(mask):
    centers = np.zeros(mask.shape, dtype=bool)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                           & mask[2:, 1:-1] & mask[1:-1, :-2]
                           & mask[1:-1, 2:])
    return centers


def _plus_mask(mask):
    """All five cells of every complete plus, detected simultaneously."""
    centers = _plus_centers(mask)
    answer = centers.copy()
    answer[:-1] |= centers[1:]
    answer[1:] |= centers[:-1]
    answer[:, :-1] |= centers[:, 1:]
    answer[:, 1:] |= centers[:, :-1]
    return answer


def _bits(mask):
    return sum(1 << int(i) for i in np.flatnonzero(mask))


def _relations(masks):
    bits = [_bits(m) for m in masks]
    locations = {}
    for index, value in enumerate(bits):
        locations.setdefault(value, []).append(index)
    relations = []
    for a in range(len(bits)):
        for b in range(a + 1, len(bits)):
            # Neither operand may be redundant, including empty/duplicate masks.
            if bits[a] & ~bits[b] and bits[b] & ~bits[a]:
                for c in locations.get(bits[a] | bits[b], ()):
                    if c not in (a, b):
                        relations.append((a, b, c))
    return sorted(relations)


def _signature(masks, relation):
    a, b, c = relation
    return tuple(sorted((_bits(masks[a]), _bits(masks[b])))), _bits(masks[c])


def _choose(masks):
    relations = _relations(masks)
    if not relations:
        raise ValueError("no nonredundant union relation")
    if len({_signature(masks, t) for t in relations}) != 1:
        raise ValueError("genuinely competing mask-level union relations")
    # Duplicate operands/results are equivalent choices; first (a,b,c), a<b.
    return relations[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    rows = list(map(int, np.flatnonzero(np.all(grid == DIVIDER, axis=1))))
    cols = list(map(int, np.flatnonzero(np.all(grid == DIVIDER, axis=0))))
    if (len(rows) < 2 or len(cols) < 2 or rows[0] != 0 or cols[0] != 0
            or rows[-1] != grid.shape[0] - 1 or cols[-1] != grid.shape[1] - 1):
        raise ValueError("complete divider border and gutters required")
    heights, widths = np.diff(rows) - 1, np.diff(cols) - 1
    if (not np.all(heights == heights[0]) or not np.all(widths == widths[0])
            or heights[0] < 3 or widths[0] < 3):
        raise ValueError("motifs must be equal and at least 3 by 3")
    slots = [(slice(r + 1, s), slice(c + 1, d))
             for r, s in zip(rows, rows[1:]) for c, d in zip(cols, cols[1:])]
    if len(slots) < 3:
        raise ValueError("at least three motifs required")
    interiors = np.concatenate([grid[slot].ravel() for slot in slots])
    colors, counts = np.unique(interiors, return_counts=True)
    if (len(colors) != 2 or any(int(c) not in PALETTE for c in colors)
            or counts[0] == counts[1]):
        raise ValueError("two interior colors with a strict majority field required")
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    raw = [grid[slot] == foreground for slot in slots]
    masks = [_plus_mask(mask) for mask in raw]
    return grid, slots, raw, masks, background, foreground, len(rows) - 1, len(cols) - 1


def _paint(grid, slots, masks, relation):
    answer = grid.copy()
    a, b, c = relation
    for index, color in ((a, OPERAND), (b, OPERAND), (c, RESULT)):
        block = answer[slots[index]]
        block[masks[index]] = color
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: visible dividers -> local plus masks -> strict union roles."""
    grid, slots, _, masks, *_ = _parse(input_grid)
    return _paint(grid, slots, masks, _choose(masks))


def _stamp(shape, centers):
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        mask[r, c] = True
        for dr, dc in CARDINALS:
            mask[r + dr, c + dc] = True
    return mask


def _adjacent(mask):
    result = np.zeros_like(mask)
    result[:-1] |= mask[1:]
    result[1:] |= mask[:-1]
    result[:, :-1] |= mask[:, 1:]
    result[:, 1:] |= mask[:, :-1]
    return result


def _decorate(rng, base, budget, attached=False):
    """Add only foreground that leaves the finished plus-union unchanged."""
    mask = base.copy()
    if attached:
        sites = np.argwhere(_adjacent(base) & ~base)
        for index in rng.permutation(len(sites)):
            point = tuple(sites[int(index)])
            mask[point] = True
            if np.array_equal(_plus_mask(mask), base):
                break
            mask[point] = False
        else:
            return None
    # Short strokes, bent strokes and isolated pixels, with no protected location.
    for _ in range(12):
        if np.count_nonzero(mask & ~base) >= budget:
            break
        r, c = int(rng.integers(mask.shape[0])), int(rng.integers(mask.shape[1]))
        direction = int(rng.integers(4))
        for _ in range(int(rng.integers(1, 5))):
            if not mask[r, c]:
                mask[r, c] = True
                if not np.array_equal(_plus_mask(mask), base):
                    mask[r, c] = False
            if np.count_nonzero(mask & ~base) >= budget:
                break
            if rng.random() < .35:
                direction = (direction + 1) % 4
            dr, dc = CARDINALS[direction]
            r, c = r + dr, c + dc
            if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]):
                break
    return mask


def _true_masks(rng, shape, simple=False, overlap=False):
    sites = [(r, c) for r in range(1, shape[0] - 1)
             for c in range(1, shape[1] - 1)]
    order = [sites[int(i)] for i in rng.permutation(len(sites))]
    if simple:
        return _stamp(shape, order[:1]), _stamp(shape, order[1:2])
    common_count = int(rng.integers(0, 3))
    common = order[:common_count]
    n_a, n_b = map(int, rng.integers(1, 4, size=2))
    a_centers = common + order[common_count:common_count + n_a]
    b_centers = common + order[common_count + n_a:common_count + n_a + n_b]
    if overlap:
        # Adjacent centers distinguish original-input detection from destructive scan.
        r, c = a_centers[0]
        neighbors = [(r + dr, c + dc) for dr, dc in CARDINALS
                     if 0 < r + dr < shape[0] - 1 and 0 < c + dc < shape[1] - 1]
        a_centers.append(neighbors[int(rng.integers(len(neighbors)))])
    return _stamp(shape, a_centers), _stamp(shape, b_centers)


def _counterfeit(rng, shape):
    """Raw A|B=C whose overlay completes a new plus: P(A)|P(B) != P(C)."""
    sites = [(r, c) for r in range(1, shape[0] - 1)
             for c in range(1, shape[1] - 1)]
    a_center, b_center, emergent = [sites[int(i)] for i in rng.choice(len(sites), 3, replace=False)]
    a, b = _stamp(shape, [a_center]), _stamp(shape, [b_center])
    r, c = emergent
    points = [(r, c)] + [(r + dr, c + dc) for dr, dc in CARDINALS]
    for index, point_index in enumerate(rng.permutation(5)):
        (a if index % 2 else b)[points[int(point_index)]] = True
    return [a, b, a | b]


def _distractor(rng, shape, targets):
    sites = [(r, c) for r in range(1, shape[0] - 1)
             for c in range(1, shape[1] - 1)]
    kind = int(rng.integers(5))
    if kind == 0:
        # A no-plus near miss, line, ring or diagonal X can remain unchanged.
        base = np.zeros(shape, dtype=bool)
    elif kind == 1:
        base = targets[int(rng.integers(3))].copy()
        r, c = sites[int(rng.integers(len(sites)))]
        base |= _stamp(shape, [(r, c)])
    else:
        amount = int(rng.integers(1, min(8, len(sites)) + 1))
        base = _stamp(shape, [sites[int(i)] for i in rng.choice(len(sites), amount, replace=False)])
    return _decorate(rng, base, int(rng.integers(1, 9)))


def _orientation(relation, columns):
    a, b = relation[:2]
    ra, ca = divmod(a, columns)
    rb, cb = divmod(b, columns)
    return "H" if ra == rb else "V" if ca == cb else "D"


def _assemble(raw, shape, layout, palette):
    h, w = shape
    rows, columns = layout
    background, foreground = palette
    grid = np.full((rows * (h + 1) + 1, columns * (w + 1) + 1), DIVIDER, dtype=np.int8)
    for index, mask in enumerate(raw):
        r, c = divmod(index, columns)
        block = grid[1 + r * (h + 1):(r + 1) * (h + 1),
                     1 + c * (w + 1):(c + 1) * (w + 1)]
        block[:] = np.where(mask, foreground, background)
    return grid


def _make_pair(rng, layout, palette, *, raw_witness=False, overlap=False,
               duplicate=False, orientation=None):
    rows, columns = layout
    for _ in range(400):
        # Canvas size is checked before allocating; largest axis is 29.
        limits = (min(11, 29 // rows - 1), min(11, 29 // columns - 1))
        shape = tuple(int(rng.integers(5, limit + 1)) for limit in limits)
        a, b = _true_masks(rng, shape, simple=raw_witness, overlap=overlap)
        if not (np.any(a & ~b) and np.any(b & ~a)):
            continue
        targets = [a, b, a | b]
        raw = [_decorate(rng, target, int(rng.integers(1, 8)), attached=True)
               for target in targets]
        if any(mask is None for mask in raw):
            continue
        if raw_witness:
            decoys = _counterfeit(rng, shape)
            derived_decoys = [_plus_mask(m) for m in decoys]
            if (not all(m.any() for m in derived_decoys)
                    or np.array_equal(derived_decoys[0] | derived_decoys[1], derived_decoys[2])
                    or max(int(m.sum()) for m in derived_decoys) <= int(targets[2].sum())):
                continue
            raw.extend(decoys)
        elif duplicate:
            # Same derived operand, independently varied non-plus camouflage.
            raw.append(_decorate(rng, a, int(rng.integers(1, 9))))
        while len(raw) < rows * columns:
            raw.append(_distractor(rng, shape, targets))
        if sum(int(m.sum()) for m in raw) * 2 >= len(raw) * shape[0] * shape[1]:
            continue
        raw = [raw[int(i)] for i in rng.permutation(len(raw))]
        masks = [_plus_mask(m) for m in raw]
        try:
            relation = _choose(masks)
        except ValueError:
            continue
        if _signature(masks, relation) != (tuple(sorted((_bits(a), _bits(b)))), _bits(a | b)):
            continue
        if orientation is not None and _orientation(relation, columns) != orientation:
            continue
        unchosen = [i for i in range(len(raw)) if i not in relation]
        if not any(masks[i].any() for i in unchosen):
            continue
        if overlap and not any(_overlap_sensitive(raw[i]) for i in relation):
            continue
        if duplicate and len(_relations(masks)) < 2:
            continue
        if not duplicate and len(_relations(masks)) > 1:
            continue
        if raw_witness:
            try:
                raw_relation = _choose(raw)
            except ValueError:
                continue
            if raw_relation == relation:
                continue
        inp = _assemble(raw, shape, layout, palette)
        parsed = _parse(inp)
        return {"input": inp, "output": _paint(inp, parsed[1], masks, relation)}
    raise RuntimeError("bounded motif construction exhausted")


def _key(grid):
    array = np.asarray(grid, dtype=np.uint8)
    return array.shape, array.tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a whole puzzle reproducibly with at least two informative demos."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    n_train = _count(num_train, "num_train", 2, 10)
    n_test = _count(num_test, "num_test", 1, 4)
    if n_train <= n_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(PALETTE, size=2, replace=False)))
    variable_palette = bool(rng.random() < .65)
    ties = bool(rng.random() < .5)
    orientations = [str(x) for x in rng.permutation(("H", "V", "D"))]
    # The held-out axis is slot count, not a hard-coded location or orientation.
    test_layout = ((2, 4), (4, 2), (3, 3))[int(rng.integers(3))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            pair_palette = (palette[::-1] if variable_palette and
                            ((split == "train" and index == 1) or (index > 1 and rng.random() < .5))
                            else palette)
            layout = ((2, 3), (3, 2))[int(rng.integers(2))] if split == "train" else test_layout
            witness = split == "train" and index == 0
            duplicate = ties and not witness and (index == 1 or rng.random() < .4)
            for _ in range(12):
                pair = _make_pair(rng, layout, pair_palette, raw_witness=witness,
                                  overlap=split == "train" and index == 1,
                                  duplicate=duplicate,
                                  orientation=orientations[index] if split == "train" and index < 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded input uniqueness attempts exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(n_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _overlap_sensitive(mask):
    """Diagnostic alternative: destructive row-major plus recoloring."""
    remaining = mask.copy()
    changed = np.zeros_like(mask)
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            points = [(r, c)] + [(r + dr, c + dc) for dr, dc in CARDINALS]
            if all(remaining[p] for p in points):
                for p in points:
                    remaining[p] = False
                    changed[p] = True
    return not np.array_equal(changed, _plus_mask(mask))


def _features(inp):
    grid, slots, raw, masks, background, foreground, rows, cols = _parse(inp)
    relation = _choose(masks)
    a, b, c = relation
    try:
        raw_relation = _choose(raw)
    except ValueError:
        raw_relation = None
    return dict(shape=tuple(grid.shape), motif_shape=tuple(raw[0].shape),
                layout=(rows, cols), palette=(background, foreground), relation=relation,
                orientation=_orientation(relation, cols),
                raw_witness=raw_relation is not None and raw_relation != relation,
                tie=len(_relations(masks)) > 1,
                selective=any(np.any(raw[i] & ~masks[i] & _adjacent(masks[i])) for i in relation),
                overlap=any(_overlap_sensitive(raw[i]) for i in relation),
                larger_decoy=any(masks[i].sum() > masks[c].sum() for i in range(len(masks)) if i not in relation),
                plus_counts=tuple(int(_plus_centers(m).sum()) for m in raw),
                shared=int((masks[a] & masks[b]).sum()),
                unique=(int((masks[a] & ~masks[b]).sum()), int((masks[b] & ~masks[a]).sum())),
                empty=sum(not m.any() for m in masks))


def validate_generated_puzzle(puzzle):
    """Check semantics, selective preservation and demonstration witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    features, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            grid, slots, raw, masks, _, foreground, _, _ = _parse(inp)
            relation = _choose(masks)
            a, b, c = relation
            assert np.any(masks[a] & ~masks[b]) and np.any(masks[b] & ~masks[a])
            assert np.array_equal(masks[a] | masks[b], masks[c])
            assert all(np.array_equal(_plus_mask(m), m) for m in masks)
            expected = _paint(grid, slots, masks, relation)
            assert np.array_equal(out, expected), "wrong selection, plus cells or colors"
            assert np.array_equal(solve(inp.copy()), out)
            changes = np.zeros_like(inp, dtype=bool)
            for index in relation:
                changes[slots[index]] = masks[index]
            assert np.array_equal(inp != out, changes)
            assert np.all(inp[changes] == foreground)
            assert np.array_equal(inp[~changes], out[~changes])
            assert any(masks[i].any() for i in range(len(masks)) if i not in relation), "union search must leave other pluses"
            features[split].append(_features(inp))
    train = features["train"]
    assert any(f["raw_witness"] for f in train), "teach plus-derived versus raw union selection"
    assert any(f["larger_decoy"] for f in train), "distinguish largest-mask selection"
    assert any(f["selective"] for f in train), "teach partial component recoloring"
    assert any(f["overlap"] for f in train), "teach simultaneous detection"
    assert len({f["orientation"] for f in train}) >= 2
    if any(f["tie"] for f in features["test"]):
        assert any(f["tie"] for f in train), "teach duplicate precedence before testing it"
    assert {f["palette"] for f in features["test"]} <= {f["palette"] for f in train}


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds; streaming measurements, no files or execution traces."""
    observed = []
    fingerprints = set()
    a_witnesses = b_witnesses = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        assert fingerprint == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fingerprint)
        a_in_training = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                saved = inp.copy()
                grid, slots, raw, masks, *_ = _parse(inp)
                try:
                    raw_relation = _choose(raw)
                except ValueError:
                    raw_relation = None
                if raw_relation is not None and raw_relation != _choose(masks):
                    # Omitting A changes selection, not merely which pixels
                    # inside an otherwise correctly selected trio are painted.
                    assert not np.array_equal(
                        _paint(grid, slots, raw, raw_relation), pair["output"])
                    a_witnesses += 1
                    a_in_training |= split == "train"
                # Omitting B and highlighting every plus changes distractors
                # that the relational rule must preserve.
                all_pluses = inp.copy()
                for slot, mask in zip(slots, masks):
                    all_pluses[slot][mask] = OPERAND
                assert not np.array_equal(all_pluses, pair["output"])
                b_witnesses += 1
                solve(inp)
                assert np.array_equal(inp, saved), "solve must not mutate its input"
        assert a_in_training
        observed.extend((split, _features(pair["input"]))
                        for split in ("train", "test") for pair in puzzle[split])
    assert len(fingerprints) == 42
    assert len({f["shape"] for _, f in observed}) >= 30
    assert len({f["motif_shape"] for _, f in observed}) >= 20
    assert len({f["palette"] for _, f in observed}) >= 15
    assert {f["layout"] for _, f in observed} == {(2, 3), (3, 2), (2, 4), (4, 2), (3, 3)}
    assert len({n for _, f in observed for n in f["plus_counts"]}) >= 10
    for split in ("train", "test"):
        subset = [f for s, f in observed if s == split]
        assert {f["orientation"] for f in subset} == {"H", "V", "D"}
        assert any(f["tie"] for f in subset)
        assert any(f["shared"] == 0 for f in subset)
        assert any(f["shared"] > 0 for f in subset)
        assert any(f["empty"] for f in subset)
    # Every legal count combination, including 2/1 and 10/4.
    for n_train in range(2, 11):
        for n_test in range(1, min(n_train - 1, 4) + 1):
            puzzle = generate_puzzle(104729, n_train, n_test)
            assert len(puzzle["train"]) == n_train and len(puzzle["test"]) == n_test
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Exact local behavior: all 512 binary neighborhoods, including clipped/X/T.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected = np.zeros((3, 3), dtype=bool)
        if all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))):
            expected[1, :] = True
            expected[:, 1] = True
        assert np.array_equal(_plus_mask(mask), expected)
    block = np.ones((3, 4), dtype=bool)
    assert _plus_mask(block).sum() == 8 and _overlap_sensitive(block)
    # Counterexample: raw union makes a plus from two incomplete pieces.
    x = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    y = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=bool)
    assert not (_plus_mask(x) | _plus_mask(y)).any() and _plus_mask(x | y).sum() == 5
    # Equivalent duplicate operands use the first occurrence, even with extra clutter.
    x = _stamp((5, 5), [(1, 1)])
    y = _stamp((5, 5), [(3, 3)])
    assert _choose([x, y, x | y, x.copy()]) == (0, 1, 2)
    # Full input-only solving outside the sampler's geometry: clean 3x4
    # motifs, equivalent operand/result copies and an empty distractor.
    x = _stamp((3, 4), [(1, 1)])
    y = _stamp((3, 4), [(1, 2)])
    raw = [x, y, x | y, x.copy(), (x | y).copy(), np.zeros((3, 4), dtype=bool)]
    inp = _assemble(raw, (3, 4), (2, 3), (0, 2))
    expected = inp.copy()
    expected[1:4, 1:5][x] = OPERAND
    expected[1:4, 6:10][y] = OPERAND
    expected[1:4, 11:15][x | y] = RESULT
    assert np.array_equal(solve(inp.tolist()), expected)
    # Changed divider, unselected pluses or non-plus camouflage must be rejected.
    puzzle = generate_puzzle(0, 2, 1)
    for point in ((0, 0), (1, 1)):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        out = bad["test"][0]["output"]
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted corrupted output")
    return {"seeds": 42, "pairs": len(observed), "distinct_puzzles": len(fingerprints),
            "grid_shapes": len({f["shape"] for _, f in observed}),
            "motif_shapes": len({f["motif_shape"] for _, f in observed}),
            "palettes": len({f["palette"] for _, f in observed}),
            "raw_union_counterexamples": sum(f["raw_witness"] for _, f in observed),
            "a_ablation_witnesses": a_witnesses,
            "b_ablation_witnesses": b_witnesses,
            "duplicate_cases": sum(f["tie"] for _, f in observed),
            "overlap_cases": sum(f["overlap"] for _, f in observed)}
