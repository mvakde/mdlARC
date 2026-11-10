"""Equalize anchored runs by occupied sites of a phase-normalized fabric.

Only NumPy and the standard library are used. The solver reads the frame,
anchors, colors and strokes from its argument; it never consults generation.
"""

from itertools import permutations, product
import numpy as np


_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_PERIMETER = np.ones((5, 5), dtype=bool)
_PERIMETER[1:4, 1:4] = False
_ORDERS = tuple(product((False, True), permutations(range(3)), permutations(range(3))))
_DIRECT = _ORDERS.index((False, (1, 2, 0), (1, 2, 0)))


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[2].any()
            and mask[:, 0].any() and mask[:, 2].any())


_MASKS = np.array([
    m for code in range(1, 511)
    if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                              dtype=bool).reshape(3, 3))
])
_DENSITY = _MASKS.sum(axis=(1, 2))
_CENTERS = _MASKS[:, 1, 1]


def _ordered(mask, order):
    transpose, rows, cols = _ORDERS[order]
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


_AGREEMENT = np.array([
    [np.array_equal(_ordered(m, order), _ordered(m, _DIRECT))
     for order in range(len(_ORDERS))] for m in _MASKS
])


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= s <= 30 for s in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return grid


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fabric(mask, shape):
    # Swatch placement has no influence on this fixed global phase.
    rows = (np.arange(shape[0]) + 1) % 3
    cols = (np.arange(shape[1]) + 1) % 3
    return mask[rows[:, None], cols[None, :]]


def _rays(shape, anchors, direction):
    result = []
    dr, dc = direction
    for r, c in anchors:
        lane = []
        r, c = r + dr, c + dc
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            lane.append((r, c))
            r, c = r + dr, c + dc
        result.append(lane)
    return result


def _hits(mask, shape, rays):
    field = _fabric(mask, shape)
    return [[j for j, (r, c) in enumerate(ray) if field[r, c]] for ray in rays]


def _analyze_input(input_grid):
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    b = int(colors[np.argmax(frequencies)])
    if int(frequencies.max()) * 2 <= grid.size:
        raise ValueError("input background must be a strict majority")
    frames = []
    h, w = grid.shape
    if min(h, w) >= 5:
        for color, frequency in zip(colors, frequencies):
            marker = int(color)
            if marker == b or int(frequency) < 19:
                continue
            corners = ((grid[:-4, :-4] == marker) & (grid[4:, :-4] == marker)
                       & (grid[:-4, 4:] == marker) & (grid[4:, 4:] == marker))
            for r, c in np.argwhere(corners):
                r, c = int(r), int(c)
                box = grid[r:r + 5, c:c + 5]
                if not np.all(box[_PERIMETER] == marker):
                    continue
                interior = box[1:4, 1:4]
                accents = set(int(x) for x in np.unique(interior)) - {b}
                if len(accents) != 1 or marker in accents:
                    continue
                accent = next(iter(accents))
                mask = interior == accent
                if _full_box(mask):
                    frames.append((r, c, marker, accent, mask.copy()))
    if len(frames) != 1:
        raise ValueError("input must contain exactly one framed 3x3 swatch")
    top, left, marker, accent, mask = frames[0]
    reserved = np.zeros(grid.shape, dtype=bool)
    reserved[top:top + 5, left:left + 5] = True
    if np.any((grid == accent) & ~reserved):
        raise ValueError("fabric ink occurs only in the input swatch")
    anchors = [tuple(int(v) for v in p)
               for p in np.argwhere((grid == marker) & ~reserved)]
    if len(anchors) not in (3, 5, 7):
        raise ValueError("expected three, five or seven anchor dots")
    inks = set(int(x) for x in colors) - {b, marker, accent}
    if not inks:
        raise ValueError("no stroke ink")
    candidates = []
    for direction in _DIRECTIONS:
        dr, dc = direction
        axis = 0 if dc else 1
        ordered = sorted(anchors, key=lambda p: p[axis])
        if len({p[1 - axis] for p in ordered}) != 1:
            continue
        positions = [p[axis] for p in ordered]
        if any(v - u < 2 for u, v in zip(positions, positions[1:])):
            continue
        rays = _rays(grid.shape, ordered, direction)
        lengths, stroke_colors, covered = [], [], set()
        valid = True
        for ray in rays:
            if not ray or any(reserved[r, c] for r, c in ray):
                valid = False
                break
            r, c = ray[0]
            ink = int(grid[r, c])
            if ink not in inks:
                valid = False
                break
            length = 0
            for r, c in ray:
                if int(grid[r, c]) != ink:
                    break
                covered.add((r, c))
                length += 1
            lengths.append(length)
            stroke_colors.append(ink)
        if not valid or covered != {tuple(int(v) for v in p)
                                     for p in np.argwhere(np.isin(grid, list(inks)))}:
            continue
        hits = _hits(mask, grid.shape, rays)
        counts = [sum(j < length for j in lane)
                  for lane, length in zip(hits, lengths)]
        if any(not count or length - 1 not in lane
               for count, length, lane in zip(counts, lengths, hits)):
            continue
        k = counts[len(ordered) // 2]
        if any(len(lane) < k for lane in hits):
            continue
        candidates.append(dict(grid=grid, background=b, marker=marker,
                               accent=accent, mask=mask, top=top, left=left,
                               reserved=reserved, anchors=ordered,
                               direction=direction, rays=rays, hits=hits,
                               lengths=lengths, colors=stroke_colors,
                               counts=counts, k=k, positions=positions))
    if len(candidates) != 1:
        raise ValueError("input must have a unique anchored-stroke interpretation")
    return candidates[0]


def _render(info, counts=None):
    field = _fabric(info["mask"], info["grid"].shape)
    out = np.where(field, info["accent"], info["background"]).astype(np.int8)
    out[info["reserved"]] = info["grid"][info["reserved"]]
    for r, c in info["anchors"]:
        out[r, c] = info["marker"]
    targets = [info["k"]] * len(info["anchors"]) if counts is None else counts
    for ray, hits, color, count in zip(info["rays"], info["hits"], info["colors"], targets):
        for j in hits[:count]:
            out[ray[j]] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every valid family input without a seed or example context."""
    return _render(_analyze_input(input_grid))


def _transform_point(point, shape, turns):
    r, c = point
    h, w = shape
    for _ in range(turns):
        r, c, h, w = w - 1 - c, r, w, h
    return int(r), int(c)


def _transform_box(top, left, shape, turns):
    corners = [_transform_point(p, shape, turns)
               for p in ((top, left), (top + 4, left + 4))]
    return tuple(min(p[k] for p in corners) for k in (0, 1))


def _layout(rng, count, direction_id, witness):
    if witness:
        gaps = [2, 2, 2, 9]
        if rng.integers(2):
            gaps.reverse()
    elif rng.random() < .4:
        gap = int(rng.integers(2, min(4, 24 // (count - 1)) + 1))
        gaps = [gap] * (count - 1)
    else:
        gaps = [2] * (count - 1)
        for _ in range(int(rng.integers(1, 25 - sum(gaps)))):
            gaps[int(rng.integers(len(gaps)))] += 1
    height = int(rng.integers(max(18, sum(gaps) + 1), 31))
    width = int(rng.integers(25 if witness else 20, 31))
    baseline = int(rng.integers(6, width - (18 if witness else 12)))
    offset = int(rng.integers(height - sum(gaps)))
    rows = [offset] + [offset + int(x) for x in np.cumsum(gaps)]
    canonical_anchors = [(r, baseline) for r in rows]
    # The swatch may occupy any clear location, including canvas boundaries.
    # A one-cell margin from all full lane rays keeps marker parsing explicit.
    slots = []
    for top in range(height - 4):
        for left in range(width - 4):
            if left + 4 < baseline - 1 or all(
                    not (top - 1 <= r <= top + 5) for r in rows):
                slots.append((top, left))
    if witness:
        slots = [p for p in slots if tuple(x % 3 for x in _transform_box(
            *p, (height, width), direction_id)) != (1, 1)]
    behind = [p for p in slots if p[1] + 4 < baseline - 1]
    beyond = [p for p in slots if p not in behind]
    pool = beyond if beyond and rng.random() < .4 else behind
    top, left = pool[int(rng.integers(len(pool)))]
    # Rotating only the layout provides all four cardinal directions. The
    # displayed swatch is selected afterward, in global row/column order.
    turns = direction_id
    shape = (height, width) if turns % 2 == 0 else (width, height)
    direction = (0, 1)
    for _ in range(turns):
        direction = (-direction[1], direction[0])
    anchors = [_transform_point(p, (height, width), turns) for p in canonical_anchors]
    axis = 0 if direction[1] else 1
    anchors.sort(key=lambda p: p[axis])
    swatch = _transform_box(top, left, (height, width), turns)
    rays = _rays(shape, anchors, direction)
    return shape, anchors, direction, swatch, rays


def _count_arguments(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _sample_counts(rng, count, k, capacities, mode, witness):
    result = [k] * count
    others = [int(i) for i in rng.permutation(count) if i != count // 2]
    if witness:
        for i in others:
            result[i] = 1
        i = others[0]
        result[i] = int(rng.integers(k + 1, capacities[i] + 1))
    elif mode.startswith("one_"):
        i = others[0]
        result[i] = (int(rng.integers(1, k)) if mode == "one_extend"
                     else int(rng.integers(k + 1, capacities[i] + 1)))
    else:
        for i in others:
            if rng.random() < .25:
                continue
            short = mode == "extend" or (mode == "mixed" and rng.random() < .5)
            result[i] = (int(rng.integers(1, k)) if short
                         else int(rng.integers(k + 1, capacities[i] + 1)))
        if mode == "mixed":
            result[others[0]] = int(rng.integers(1, k))
            i = others[1]
            result[i] = int(rng.integers(k + 1, capacities[i] + 1))
        elif result == [k] * count:
            i = others[0]
            result[i] = 1 if mode == "extend" else k + 1
    return result


def _make_pair(rng, palette, count, direction_id, allowed, first,
               witness=False, avoid_k=None, mode="mixed", mono=False):
    shape, anchors, direction, swatch, rays = _layout(rng, count, direction_id, witness)
    choices = []
    for mask_id in np.flatnonzero(allowed):
        mask_id = int(mask_id)
        mask = _MASKS[mask_id]
        if witness:
            attached_rows = (np.arange(3) - swatch[0] - 1) % 3
            attached_cols = (np.arange(3) - swatch[1] - 1) % 3
            attached = mask[attached_rows[:, None], attached_cols[None, :]]
            if np.array_equal(attached, _fabric(mask, (3, 3))):
                continue
        hits = _hits(mask, shape, rays)
        maximum = min(8, min(map(len, hits)) - 1)
        ks = [k for k in range(4 if witness else 2, maximum + 1) if k != avoid_k]
        if witness:
            ks = [k for k in ks
                  if len({lane[k - 1] for lane in hits}) > 1
                  and any(sum(j <= hits[count // 2][k - 1] for j in lane) != k
                          for lane in hits)]
        if first is not None and int((_AGREEMENT[mask_id] & _AGREEMENT[first]).sum()) != 1:
            continue
        if ks:
            choices.append((mask_id, hits, ks))
    if not choices:
        raise RuntimeError("no compatible swatch for layout")
    # Sample densities first, rather than overwhelming sparse motifs by their
    # combinatorial rarity. Then sample within the chosen density.
    density = int(rng.choice(sorted({_DENSITY[c[0]] for c in choices})))
    eligible = [c for c in choices if int(_DENSITY[c[0]]) == density]
    mask_id, hits, ks = eligible[int(rng.integers(len(eligible)))]
    k = int(rng.choice(ks))
    capacities = [len(lane) for lane in hits]
    counts = _sample_counts(rng, count, k, capacities, mode, witness)
    lengths = [lane[q - 1] + 1 for lane, q in zip(hits, counts)]
    background, marker, accent, ink_pool = palette
    if mono:
        colors = [int(rng.choice(ink_pool))] * count
    else:
        colors = [int(rng.choice(ink_pool)) for _ in range(count)]
        colors[0], colors[-1] = [int(x) for x in rng.choice(ink_pool, 2, replace=False)]
    inp = np.full(shape, background, dtype=np.int8)
    top, left = swatch
    box = inp[top:top + 5, left:left + 5]
    box[_PERIMETER] = marker
    box[1:4, 1:4][_MASKS[mask_id]] = accent
    for anchor, ray, length, color in zip(anchors, rays, lengths, colors):
        inp[anchor] = marker
        for p in ray[:length]:
            inp[p] = color
    # Ground truth is drawn from construction parameters, not by solve().
    out = np.where(_fabric(_MASKS[mask_id], shape), accent, background).astype(np.int8)
    out[top:top + 5, left:left + 5] = inp[top:top + 5, left:left + 5]
    for anchor, ray, lane, color in zip(anchors, rays, hits, colors):
        out[anchor] = marker
        for j in lane[:k]:
            out[ray[j]] = color
    return {"input": inp, "output": out}, mask_id, k


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count_arguments(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # Nonnegative seeds use default_rng(seed) directly. Negative integer seeds
    # have a documented uint64 interpretation rather than an RNG error.
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    shuffled = [int(c) for c in rng.permutation(10)]
    palette = (*shuffled[:3], shuffled[3:3 + int(rng.integers(2, 5))])
    regime = str(rng.choice(("center_gap", "denser", "sparser")))
    directions = [int(rng.integers(4))]
    directions.append((directions[0] + int(rng.choice((1, 3)))) % 4)
    allowed_train = np.ones(len(_MASKS), dtype=bool)
    allowed_test = allowed_train.copy()
    if regime == "center_gap":
        allowed_train &= _CENTERS
        allowed_test &= ~_CENTERS
    elif regime == "denser":
        allowed_train &= _DENSITY <= 5
        allowed_test &= _DENSITY >= 6
    else:
        allowed_train &= _DENSITY >= 5
        allowed_test &= _DENSITY <= 4
    readable = (_MASKS.any(axis=1).all(axis=1)
                & _MASKS.any(axis=2).all(axis=1))
    # First witnesses must have an available, different-density companion.
    # All companion phases are nonempty and thus fit any second layout.
    viable_first = np.array([
        np.any(allowed_train & readable & (_DENSITY != _DENSITY[j])
               & ((_AGREEMENT & _AGREEMENT[j]).sum(axis=1) == 1))
        for j in range(len(_MASKS))])
    puzzle = {"train": [], "test": []}
    used, mask_ids, first_k = set(), [], None
    modes = ("mixed", "extend", "trim", "one_extend", "one_trim")
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            witness = split == "train" and index == 0
            count = 5 if witness else int(rng.choice((3, 7))) if split == "train" and index == 1 else int(rng.choice((3, 5, 7)))
            direction = directions[index % 2] if split == "train" else int(rng.choice(directions))
            allowed = (allowed_train if split == "train" else allowed_test).copy()
            allowed[mask_ids] = False
            if split == "train" and index < 2:
                # Informative training is readable in both axial orientations.
                allowed &= readable
            if witness:
                allowed &= viable_first
            if split == "train" and index == 1:
                allowed &= _DENSITY != _DENSITY[mask_ids[0]]
            first = mask_ids[0] if split == "train" and index == 1 else None
            mode = "mixed" if split == "train" and index < 2 else str(rng.choice(modes))
            for _ in range(60):
                try:
                    pair, mask_id, k = _make_pair(
                        rng, palette, count, direction, allowed, first,
                        witness=witness, avoid_k=first_k if split == "train" and index == 1 else None,
                        mode=mode, mono=witness or (not (split == "train" and index == 1)
                                                    and rng.random() < .3))
                except RuntimeError:
                    continue
                if _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("bounded layout search exhausted")
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            mask_ids.append(mask_id)
            if witness:
                first_k = k
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze_input(inp)
    assert inp.shape == out.shape
    field = _fabric(info["mask"], inp.shape)
    excluded = info["reserved"].copy()
    assert np.array_equal(out[excluded], inp[excluded]), "swatch changed"
    for p in info["anchors"]:
        assert out[p] == info["marker"], "anchor changed"
        excluded[p] = True
    for ray, hits, color in zip(info["rays"], info["hits"], info["colors"]):
        selected = {ray[j] for j in hits[:info["k"]]}
        assert len(selected) == info["k"]
        for p in ray:
            expected = color if p in selected else info["accent"] if field[p] else info["background"]
            assert out[p] == expected, "wrong stitch count, phase or free endpoint"
            excluded[p] = True
    assert np.array_equal(out[~excluded], np.where(field, info["accent"], info["background"])[~excluded]), "fabric corrupted"
    # Carrier gaps survive even inside highlighted regions.
    assert np.all(out[~field & ~info["reserved"] & (out != info["marker"])] == info["background"])
    assert not np.array_equal(inp, out)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count_arguments(len(puzzle["train"]), len(puzzle["test"]))
    used, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in used, "duplicate input"
            used.add(key)
            info = _validate_pair(pair)
            assert any(q != info["k"] for q in info["counts"]), "sample has no event-count repair"
            records[split].append(info)
    train = records["train"]
    conventions = {(i["background"], i["marker"], i["accent"])
                   for s in records.values() for i in s}
    assert len(conventions) == 1, "puzzle-wide roles changed"
    assert len({len(i["anchors"]) for i in train}) >= 2
    assert len({i["k"] for i in train}) >= 2
    assert len({int(i["mask"].sum()) for i in train}) >= 2
    assert any(len(set(i["colors"])) == 1 for i in train)
    assert any(len(set(i["colors"])) > 1 for i in train)
    assert any(min(i["counts"]) < i["k"] < max(i["counts"]) for i in train)
    assert any(sorted(i["counts"])[len(i["counts"]) // 2] != i["k"] for i in train)
    assert any(sorted(i["lengths"])[len(i["lengths"]) // 2] != i["lengths"][len(i["lengths"]) // 2] for i in train)
    assert any(len({lane[i["k"] - 1] for lane in i["hits"]}) > 1 for i in train)
    assert any(_geometric_witness(i) for i in train)
    agree = np.ones(len(_ORDERS), dtype=bool)
    for i in train:
        agree &= np.array([np.array_equal(_ordered(i["mask"], j), _ordered(i["mask"], _DIRECT))
                           for j in range(len(_ORDERS))])
    assert np.flatnonzero(agree).tolist() == [_DIRECT], "phase not demonstrated"
    # At least one observable center or density shift, without latent metadata.
    test = records["test"]
    center_shift = all(i["mask"][1, 1] for i in train) and all(not i["mask"][1, 1] for i in test)
    train_density = [int(i["mask"].sum()) for i in train]
    test_density = [int(i["mask"].sum()) for i in test]
    assert center_shift or min(test_density) > max(train_density) or max(test_density) < min(train_density)


def _geometric_witness(info):
    positions = np.asarray(info["positions"])
    distances = abs(2 * positions - positions[0] - positions[-1])
    nearest = np.flatnonzero(distances == distances.min())
    return (len(nearest) == 1 and int(nearest[0]) != len(positions) // 2
            and info["counts"][int(nearest[0])] != info["k"])


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> dict:
    """Bounded semantic, counterfactual, variation and API checks."""
    masks, shapes, directions, counts, references, palettes, structures = (set() for _ in range(7))
    modes, centers, densities, empty_axes, fingerprints = set(), set(), set(), 0, set()
    rng = np.random.default_rng(1937)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 3:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _validate_pair(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                masks.add(info["mask"].tobytes())
                shapes.add(pair["input"].shape)
                directions.add(info["direction"])
                counts.add(len(info["anchors"]))
                references.add(info["k"])
                palettes.add((info["background"], info["marker"], info["accent"]))
                structures.add((info["mask"].tobytes(), tuple(info["positions"]),
                                tuple(info["lengths"]), info["k"], info["direction"]))
                centers.add(bool(info["mask"][1, 1]))
                densities.add(int(info["mask"].sum()))
                empty_axes += int(not info["mask"].any(axis=0).all() or not info["mask"].any(axis=1).all())
                short, long = min(info["counts"]) < info["k"], max(info["counts"]) > info["k"]
                modes.add("mixed" if short and long else "extend" if short else "trim" if long else "same")
        pair = puzzle["train"][0]
        info = _analyze_input(pair["input"])
        # Omit A: keep the original per-lane number of events.
        assert not np.array_equal(_render(info, info["counts"]), pair["output"])
        # Keep B's texture but omit event distance: use middle physical reach.
        central_length = info["lengths"][len(info["anchors"]) // 2]
        wrong_counts = [sum(j < central_length for j in hit) for hit in info["hits"]]
        assert not np.array_equal(_render(info, wrong_counts), pair["output"])
        # A swatch-attached phase is a concrete alternative to normalization.
        rows, cols = np.indices(pair["input"].shape)
        attached = info["mask"][(rows - info["top"] - 1) % 3,
                                (cols - info["left"] - 1) % 3]
        assert not np.array_equal(attached, _fabric(info["mask"], rows.shape))
        # A reference perturbation must change every lane's stitch budget.
        altered = pair["input"].copy()
        mid = len(info["anchors"]) // 2
        ray, hits = info["rays"][mid], info["hits"][mid]
        for p in ray[hits[info["k"] - 2] + 1:info["lengths"][mid]]:
            altered[p] = info["background"]
        changed = _analyze_input(altered)
        assert changed["k"] == info["k"] - 1
        assert not np.array_equal(solve(altered), pair["output"])
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(fingerprints) == 42
    assert len(masks) >= 60 and len(shapes) >= 45 and len(structures) >= 190
    assert directions == set(_DIRECTIONS) and counts == {3, 5, 7}
    assert len(references) >= 5 and len(palettes) >= 30
    assert centers == {False, True} and densities == set(range(2, 9))
    assert {"mixed", "extend", "trim"} <= modes and empty_axes >= 2
    # Hand-built members outside the sampler's budgets and ink-pool sizes.
    # A four-corner swatch has a permanently empty phase, deliberately avoided
    # by these seven lanes. Exercise one stitch and boundary-reaching budgets.
    probe_mask = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool)
    lane_positions = (1, 5, 7, 11, 13, 17, 19)
    probe_colors = (1, 3, 4, 5, 6, 7, 9)
    for direction in _DIRECTIONS:
        dr, dc = direction
        if dc:
            shape = (21, 30)
            anchors = [(r, 6 if dc > 0 else 23) for r in lane_positions]
            top, left = (0, 0) if dc > 0 else (0, 25)
        else:
            shape = (30, 21)
            anchors = [(6 if dr > 0 else 23, c) for c in lane_positions]
            top, left = (0, 0) if dr > 0 else (25, 0)
        rays = _rays(shape, anchors, direction)
        hits = _hits(probe_mask, shape, rays)
        for budget in (1, min(map(len, hits))):
            assert budget == 1 or budget > 8
            inp = np.zeros(shape, dtype=np.int8)
            box = inp[top:top + 5, left:left + 5]
            box[_PERIMETER] = 8
            box[1:4, 1:4][probe_mask] = 2
            for i, (anchor, ray, hit, color) in enumerate(zip(anchors, rays, hits, probe_colors)):
                inp[anchor] = 8
                raw_count = budget if i == 3 else 1
                for p in ray[:hit[raw_count - 1] + 1]:
                    inp[p] = color
            result = solve(inp)
            assert np.array_equal(result[top:top + 5, left:left + 5], box)
            for anchor, ray, hit, color in zip(anchors, rays, hits, probe_colors):
                assert result[anchor] == 8
                colored = {p for p in ray if result[p] == color}
                assert colored == {ray[j] for j in hit[:budget]}
                assert all(result[p] == 0 for p in ray if not probe_mask[(p[0] + 1) % 3, (p[1] + 1) % 3])
            if budget > 8 and (dc > 0 or dr > 0):
                assert any(ray[hit[budget - 1]][0] in (0, shape[0] - 1)
                           or ray[hit[budget - 1]][1] in (0, shape[1] - 1)
                           for ray, hit in zip(rays, hits))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(17, 2, 1)
    p = broken["test"][0]
    i = _analyze_input(p["input"])
    p["output"][i["rays"][0][i["hits"][0][0]]] = i["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing stitch was accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), motifs=len(masks),
                shapes=len(shapes), structures=len(structures),
                directions=len(directions), lane_counts=sorted(counts),
                reference_counts=sorted(references), densities=sorted(densities),
                palette_roles=len(palettes), empty_axis_examples=empty_axes)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
