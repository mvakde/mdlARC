"""Mirror-seal identity defines communicating, volume-conserving vessels.

Only Python's standard library and NumPy are used. The solver reads a single
input and has no access to the random generator or any demonstration labels.
"""
from collections import defaultdict
import numpy as np


BACKGROUND, LIQUID, ERASE = 0, 1, 9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(a.dtype.kind in "iu" and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        parts.append(cells)
    return parts


def _orbits(h, w):
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            yield tuple(sorted({(r, c), (h - 1 - r, c),
                                (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _repair(seal):
    restored = seal.copy()
    for orbit in _orbits(*seal.shape):
        witnesses = {int(seal[p]) for p in orbit if seal[p] != ERASE}
        _require(len(witnesses) == 1, "seal orbit is missing or contradictory")
        color = next(iter(witnesses))
        for p in orbit:
            restored[p] = color
    return restored


def _seal_key(seal):
    return seal.shape, seal.astype(np.int8).tobytes()


def _parse(a):
    """Read cap, seal and chamber from each separated solid component."""
    solids = (a != BACKGROUND) & (a != LIQUID)
    vessels, all_free, all_seals = [], np.zeros_like(solids), np.zeros_like(solids)
    for cells in _components(solids):
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells) + 1
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells) + 1
        box = a[r0:r1, c0:c1]
        h, w = box.shape
        _require(h >= 7 and w >= 5, "vessel is too small")
        wall = int(box[0, 0])
        _require(2 <= wall <= 8, "invalid vessel frame")
        _require(np.all(box[0] == wall) and np.all(box[-1] == wall)
                 and np.all(box[:, 0] == wall) and np.all(box[:, -1] == wall),
                 "vessel perimeter is not solid")
        seal_cols = np.flatnonzero(box[1] != wall)
        _require(len(seal_cols) > 0, "no seal in cap")
        left, right = int(seal_cols[0]), int(seal_cols[-1]) + 1
        _require(np.array_equal(seal_cols, np.arange(left, right))
                 and 0 < left < right < w, "seal is not framed")
        divider = 2
        while divider < h and not np.all(box[divider] == wall):
            divider += 1
        sh, sw = divider - 1, right - left
        _require(3 <= sh <= 5 and 3 <= sw <= 5 and divider < h - 1,
                 "invalid cap geometry")
        raw = box[1:divider, left:right].copy()
        _require(np.all((raw >= 2) & (raw != wall)), "invalid seal payload")
        header = box[:divider + 1].copy()
        header[1:divider, left:right] = wall
        _require(np.all(header == wall), "unexpected cell outside seal")
        repaired = _repair(raw)
        local_free = np.zeros_like(box, dtype=bool)
        local_free[divider + 1:-1, 1:-1] = (
            (box[divider + 1:-1, 1:-1] == BACKGROUND)
            | (box[divider + 1:-1, 1:-1] == LIQUID))
        # The intended family has grounded column floors, not hidden cavities.
        for c in range(1, w - 1):
            column = local_free[divider + 1:-1, c]
            _require(column.any() and not np.any(np.diff(column.astype(int)) > 0),
                     "chamber floor must be grounded")
        chamber = box[divider + 1:, :]
        _require(np.all((chamber == wall) | (chamber == BACKGROUND)
                        | (chamber == LIQUID)), "unexpected chamber color")
        coords = np.argwhere(local_free) + np.array([r0, c0])
        seal_slice = (slice(r0 + 1, r0 + divider), slice(c0 + left, c0 + right))
        all_free[r0:r1, c0:c1] |= local_free
        all_seals[seal_slice] = True
        vessels.append({"box": (r0, r1, c0, c1), "wall": wall,
                        "seal_slice": seal_slice, "raw": raw, "seal": repaired,
                        "key": _seal_key(repaired), "coords": coords,
                        "volume": int(np.count_nonzero(box[local_free] == LIQUID))})
    _require(bool(vessels), "no vessels")
    _require(np.all((a != LIQUID) | all_free), "liquid outside chambers")
    _require(np.all((a != ERASE) | all_seals), "erasure outside seals")
    # A shared wall color is visibly encoded by every perimeter.
    _require(len({v["wall"] for v in vessels}) == 1, "inconsistent example frames")
    for i, v in enumerate(vessels):
        r0, r1, c0, c1 = v["box"]
        for u in vessels[:i]:
            a0, a1, b0, b1 = u["box"]
            _require(r1 < a0 or a1 < r0 or c1 < b0 or b1 < c0,
                     "vessel boxes need an exterior gap")
    return vessels, all_free, all_seals


def _groups(vessels, mode="circuits"):
    grouped = defaultdict(list)
    for i, v in enumerate(vessels):
        key = v["key"] if mode == "circuits" else (
            i if mode == "independent" else _seal_key(v["raw"]) if mode == "raw" else 0)
        grouped[key].append(i)
    return list(grouped.values())


def _fill(coords, volume):
    """Complete absolute rows only; order within a row is immaterial."""
    filled = np.zeros(len(coords), dtype=bool)
    _require(0 <= volume <= len(coords), "liquid exceeds circuit capacity")
    for row in sorted(set(map(int, coords[:, 0])), reverse=True):
        level = coords[:, 0] == row
        capacity = int(level.sum())
        if volume == 0:
            break
        _require(volume >= capacity, "liquid would partially fill a circuit level")
        filled |= level
        volume -= capacity
    _require(volume == 0, "unallocated liquid")
    return coords[filled]


def _render(a, vessels, free, mode="circuits"):
    out = a.copy()
    out[free] = BACKGROUND
    for v in vessels:
        out[v["seal_slice"]] = v["seal"]
    for indices in _groups(vessels, mode):
        coords = np.concatenate([vessels[i]["coords"] for i in indices])
        water = _fill(coords, sum(vessels[i]["volume"] for i in indices))
        if len(water):
            out[water[:, 0], water[:, 1]] = LIQUID
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair seals, infer circuits, and conserve/fill their visible liquid."""
    a = _grid(input_grid)
    vessels, free, _ = _parse(a)
    return _render(a, vessels, free)


def _different_ablation(a, vessels, free, expected, mode):
    try:
        alternative = _render(a, vessels, free, mode)
    except ValueError:
        return True
    return not np.array_equal(alternative, expected)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer))
                 and not isinstance(value, (bool, np.bool_)) and lo <= value <= hi,
                 f"illegal {name}")
    nt, ns = int(num_train), int(num_test)
    _require(nt > ns, "num_train must exceed num_test")
    return nt, ns


def _make_codes(rng, h, w, wall, count):
    orbits = list(_orbits(h, w))
    equal_size = defaultdict(list)
    for i, orbit in enumerate(orbits):
        if (0, 0) not in orbit and len(orbit) > 1:
            equal_size[len(orbit)].append(i)
    interchangeable = max(equal_size.values(), key=len)
    _require(len(interchangeable) >= count, "seal too small for distinct circuit codes")
    colors = rng.choice([c for c in range(2, 9) if c != wall],
                        size=int(rng.integers(max(3, len(interchangeable)), 7)), replace=False)
    values = rng.choice(colors, size=len(orbits))
    # Exactly the same color multiset and corner orbit in every circuit code.
    values[interchangeable] = colors[:len(interchangeable)]
    result = []
    for shift in range(count):
        variant = values.copy()
        variant[interchangeable] = np.roll(values[interchangeable], shift)
        code = np.empty((h, w), dtype=np.int8)
        for color, orbit in zip(variant, orbits):
            for p in orbit:
                code[p] = color
        result.append(code)
    return result


def _damage(rng, code, harder):
    h, w = code.shape
    family = int(rng.integers(4))
    if family == 0:
        mask = rng.random((h, w)) < rng.uniform(.12, .45)
    else:
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            dh = 1 if family == 1 else int(rng.integers(1, h + 1))
            dw = 1 if family == 2 else int(rng.integers(1, w + 1))
            mask[r:min(h, r + dh), c:min(w, c + dw)] = True
    orbits = list(_orbits(h, w))
    four = [o for o in orbits if len(o) == 4]
    # Every seal retains local witnesses, including singletons on both axes.
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    if harder:
        for j in rng.permutation(len(four))[:min(len(four), 1 + int(rng.integers(2)))]:
            orbit = four[int(j)]
            witness = orbit[int(rng.integers(4))]
            for p in orbit:
                mask[p] = p != witness
    if not mask.any():
        orbit = next(o for o in orbits if len(o) > 1)
        mask[orbit[0]] = True
    raw = code.copy()
    raw[mask] = ERASE
    return raw


def _distinct_witness_mask(code, used):
    """Four deterministic, locally recoverable masks for up to three partners."""
    corner = next(o for o in _orbits(*code.shape) if (0, 0) in o)
    for witness in corner:
        raw = code.copy()
        for p in corner:
            if p != witness:
                raw[p] = ERASE
        if _seal_key(raw) not in used:
            return raw
    raise ValueError("corner-witness masks exhausted")


def _floor_profile(rng, width, depth):
    style = int(rng.integers(6))
    x = np.arange(width)
    if style == 0:
        rise = np.zeros(width, dtype=int)
    elif style == 1:
        rise = (x >= int(rng.integers(1, width))) * int(rng.integers(1, depth))
    elif style == 2:
        rise = x * (depth - 1) // max(1, width - 1)
    elif style == 3:
        rise = abs(2 * x - width + 1) * (depth - 1) // max(1, width - 1)
    elif style == 4:
        rise = rng.integers(0, depth, size=(width + 1) // 2).repeat(2)[:width]
    else:
        rise = rng.integers(0, depth, size=width)
    if rng.random() < .5:
        rise = rise[::-1].copy()
    rise -= rise.min()
    return rise


def _pack(rng, sizes, bands):
    """Varied shelf packing: each band has independent gaps and elevations."""
    groups = [list(np.flatnonzero(np.asarray(bands) == k)) for k in sorted(set(bands))]
    row_heights = [max(sizes[i][0] for i in ids) for ids in groups]
    row_widths = [sum(sizes[i][1] for i in ids) + len(ids) - 1 for ids in groups]
    base_h, base_w = sum(row_heights) + len(groups) - 1, max(row_widths)
    h = int(rng.integers(base_h, 31))
    w = int(rng.integers(base_w, 31))
    # Allocate spare vertical space among exterior gaps; not a fixed border.
    vextra = rng.multinomial(h - base_h, np.ones(len(groups) + 1) / (len(groups) + 1))
    positions = [None] * len(sizes)
    row = int(vextra[0])
    for k, ids in enumerate(groups):
        order = list(rng.permutation(ids))
        extra = rng.multinomial(w - row_widths[k], np.ones(len(ids) + 1) / (len(ids) + 1))
        col = int(extra[0])
        for j, i in enumerate(order):
            bh, bw = sizes[i]
            # A band is a packing device: boxes have independent y offsets.
            offset = int(rng.integers(row_heights[k] - bh + 1))
            positions[i] = (row + offset, col)
            col += bw + 1 + int(extra[j + 1])
        row += row_heights[k] + 1 + int(vextra[k + 1])
    return (h, w), positions


def _target_options(vessels, indices):
    coords = np.concatenate([vessels[i]["coords"] for i in indices])
    options = []
    for level in sorted(set(map(int, coords[:, 0]))):
        target = coords[coords[:, 0] >= level]
        if not 0 < len(target) < len(coords):
            continue
        # A strictly higher dry cell in another vessel can donate liquid.
        exchanges = []
        for i in indices:
            donor = vessels[i]["coords"]
            donor = donor[donor[:, 0] < level]
            for j in indices:
                if i == j or not len(donor):
                    continue
                receiver = vessels[j]["coords"]
                receiver = receiver[receiver[:, 0] >= level]
                if len(receiver):
                    exchanges.append((i, j, donor, receiver))
        options.append((level, target, exchanges))
    return options


def _local_donor(vessels, indices, option):
    """A whole locally stable pool that drains into its lower circuit partner."""
    level, target, _ = option
    for i in indices:
        coords = vessels[i]["coords"]
        if np.all(coords[:, 0] < level):
            try:
                return _fill(coords, len(target))
            except ValueError:
                pass
    return None


def _supply(rng, vessels, indices, target, exchanges):
    coords = np.concatenate([vessels[i]["coords"] for i in indices])
    volume = len(target)
    style = int(rng.integers(5))
    anchor = coords[int(rng.integers(len(coords)))]
    dr, dc = abs(coords[:, 0] - anchor[0]), abs(coords[:, 1] - anchor[1])
    scores = rng.random(len(coords))
    if style == 0:
        scores += dr + dc
    elif style == 1:
        scores += 3 * dr + .1 * dc
    elif style == 2:
        scores += 3 * dc + .1 * dr
    elif style == 3:
        scores += coords[:, 0] * 2
    else:
        target_set = set(map(tuple, target))
        scores += np.array([0 if tuple(p) in target_set else 3 for p in coords])
    source = set(map(tuple, coords[np.argsort(scores)[:volume]]))
    if exchanges:
        # Set at least one cross-vessel count imbalance, even for partly-settled sources.
        def counts(points):
            return [sum(tuple(p) in points for p in vessels[i]["coords"]) for i in indices]
        if counts(source) == counts(set(map(tuple, target))):
            _, _, donors, receivers = exchanges[int(rng.integers(len(exchanges)))]
            donor = next((tuple(p) for p in donors if tuple(p) not in source), None)
            receiver = next((tuple(p) for p in receivers if tuple(p) in source), None)
            if donor is not None and receiver is not None:
                source.remove(receiver)
                source.add(donor)
            else:
                # A known target-to-source transfer is always available.
                source = set(map(tuple, target))
                source.remove(tuple(receivers[int(rng.integers(len(receivers)))]))
                source.add(tuple(donors[int(rng.integers(len(donors)))]))
    elif source == set(map(tuple, target)):
        outside = [tuple(p) for p in coords if tuple(p) not in source]
        if outside:
            source.remove(max(source))
            source.add(outside[int(rng.integers(len(outside)))])
    return np.array(sorted(source), dtype=int)


def _make_pair(rng, kind, _emergency=False):
    """kind 0: shared level, 1: high-to-low circuits, 2: varied, 3: test triple."""
    for attempt in ([79] if _emergency else range(80)):
        emergency = attempt == 79
        n = 3 if kind == 0 else 4 if kind == 1 else int(rng.integers(3, 5)) if kind == 2 else int(rng.integers(4, 7))
        if emergency:
            n = 3 if kind == 0 else 4
        circuit_count = 3 if n == 6 and rng.random() < .45 else 2
        sh, sw = map(int, rng.integers(3 if circuit_count == 2 else 4, 6, size=2))
        wall = int(rng.integers(2, 9))
        codes = _make_codes(rng, sh, sw, wall, circuit_count)
        sizes = [(sh + int(rng.integers(3, 6)) + 3,
                  sw + 2 + int(rng.integers(0, 2))) for _ in range(n)]
        if emergency:
            sizes = [(sh + 7, sw + 2)] * n  # Four flat chamber rows.
        # Three-vessel shared-level training is spread over one band; other
        # examples expose changes in absolute elevation as well as width/depth.
        bands = [0] * n if kind == 0 else [0] * ((n + 1) // 2) + [1] * (n // 2)
        shape, positions = _pack(rng, sizes, bands)
        classes = [0] * n
        if kind == 0:
            order = sorted(range(n), key=lambda i: positions[i][1])
            classes[order[1]] = 1  # Linked endpoints surround an unrelated vessel.
        elif kind == 1:
            classes = [0, 1, 0, 1]  # Each circuit spans the two elevation bands.
        elif kind == 3:
            lower = list(np.flatnonzero(np.asarray(bands) == 1))
            upper = list(np.flatnonzero(np.asarray(bands) == 0))
            linked = list(rng.choice(lower, size=2, replace=False)) + [int(rng.choice(upper))]
            classes = [1] * n
            for i in linked:
                classes[i] = 0
            if circuit_count == 3:
                rest = [i for i in range(n) if i not in linked]
                classes[rest[int(rng.integers(len(rest)))]] = 2
        else:
            order = list(rng.permutation(n))
            classes = [1] * n
            for i in order[:2]:
                classes[i] = 0
            if emergency:
                classes = [0, 0, 1, 1]
        a = np.zeros(shape, dtype=np.int8)
        used_raw = defaultdict(set)
        for i, ((bh, bw), (r, c)) in enumerate(zip(sizes, positions)):
            block = np.full((bh, bw), wall, dtype=np.int8)
            left = int(rng.integers(1, bw - sw))
            for _ in range(40):
                raw = _damage(rng, codes[classes[i]], harder=(i == 0 or kind == 3 or rng.random() < .4))
                key = _seal_key(raw)
                if key not in used_raw[classes[i]]:
                    used_raw[classes[i]].add(key)
                    break
            else:
                # Four distinct corner-witness masks suffice for a group of
                # at most three; this removes the rare random-mask failure.
                raw = _distinct_witness_mask(codes[classes[i]], used_raw[classes[i]])
                used_raw[classes[i]].add(_seal_key(raw))
            block[1:sh + 1, left:left + sw] = raw
            chamber_start = sh + 2
            depth = bh - chamber_start - 1
            profile = (np.zeros(bw - 2, dtype=int) if emergency else
                       _floor_profile(rng, bw - 2, depth))
            for col, rise in enumerate(profile, 1):
                block[chamber_start:bh - 1 - int(rise), col] = BACKGROUND
            a[r:r + bh, c:c + bw] = block
        vessels, free, _ = _parse(a)
        targets, valid = [], True
        for circuit_index, ids in enumerate(_groups(vessels)):
            options = _target_options(vessels, ids)
            if len(ids) >= 2:
                options = [o for o in options if o[2]]
            if kind == 0 and len(ids) == 2:
                options = [o for o in options if all(np.any(vessels[i]["coords"][:, 0] >= o[0]) for i in ids)]
            if kind == 1:
                options = [o for o in options if sum(np.any(vessels[i]["coords"][:, 0] >= o[0]) for i in ids) == 1
                           and _local_donor(vessels, ids, o) is not None]
            if kind == 3 and vessels[ids[0]]["key"] == _seal_key(codes[0]):
                options = [o for o in options if sum(np.any(vessels[i]["coords"][:, 0] >= o[0]) for i in ids) >= 2
                           and any(np.all(vessels[i]["coords"][:, 0] < o[0]) for i in ids)]
            if emergency:
                bottom = max(int(vessels[i]["coords"][:, 0].max()) for i in ids)
                depth = (int(len(ids) == 1) if kind in (0, 3) else circuit_index)
                options = [o for o in options if o[0] == bottom - depth]
            if not options:
                valid = False
                break
            level, target, exchanges = options[int(rng.integers(len(options)))]
            source = (_local_donor(vessels, ids, (level, target, exchanges)) if kind == 1 else
                      _supply(rng, vessels, ids, target, exchanges))
            a[source[:, 0], source[:, 1]] = LIQUID
            targets.append(target)
        if not valid:
            continue
        expected = a.copy()
        expected[free] = BACKGROUND
        for v in vessels:
            expected[v["seal_slice"]] = v["seal"]
        for target in targets:
            expected[target[:, 0], target[:, 1]] = LIQUID
        vessels, free, _ = _parse(a)
        if (np.array_equal(solve(a), expected)
                and all(_different_ablation(a, vessels, free, expected, m)
                        for m in ("independent", "global", "raw"))):
            return {"input": a, "output": expected}
    raise RuntimeError("bounded vessel construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)),
             "seed must be an integer")
    rng = np.random.default_rng(int(seed))
    nt, ns = _counts(num_train, num_test)
    kinds = [0, 1] + [2] * (nt - 2)
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", kinds), ("test", [3] * ns)):
        for kind in schedule:
            for _ in range(32):
                pair = _make_pair(rng, kind)
                key = _seal_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    vessels, free, seals = _parse(a)
    assert np.array_equal(solve(a), out), "incorrect solved output"
    assert not np.any(out == ERASE)
    immutable = ~free & ~seals
    assert np.array_equal(a[immutable], out[immutable]), "solid/exterior changed"
    assert np.array_equal(a[seals & (a != ERASE)], out[seals & (a != ERASE)])
    assert (a == LIQUID).sum() == (out == LIQUID).sum() > 0
    assert np.any((a == LIQUID) != (out == LIQUID)), "liquid did not move"
    assert np.argwhere(out == LIQUID)[:, 0].sum() > np.argwhere(a == LIQUID)[:, 0].sum()
    groups = _groups(vessels)
    histograms, codes, levels, transfer, dry, shared = [], [], [], False, 0, False
    diagonal = 0
    floor_shapes, input_components = set(), len(_components(a == LIQUID))
    for v in vessels:
        restored = v["seal"]
        assert np.array_equal(restored, restored[::-1]) and np.array_equal(restored, restored[:, ::-1])
        for orbit in _orbits(*restored.shape):
            assert any(v["raw"][p] != ERASE for p in orbit)
            if len(orbit) == 4 and sum(v["raw"][p] == ERASE for p in orbit) == 3:
                diagonal += 1
        coords = v["coords"]
        r0, r1, c0, c1 = v["box"]
        floors = tuple(int(coords[coords[:, 1] == c, 0].max() - r0)
                       for c in range(c0 + 1, c1 - 1))
        floor_shapes.add(floors)
    for ids in groups:
        coords = np.concatenate([vessels[i]["coords"] for i in ids])
        before = sum(vessels[i]["volume"] for i in ids)
        water = coords[out[coords[:, 0], coords[:, 1]] == LIQUID]
        assert len(water) == before and 0 < before < len(coords)
        assert np.array_equal(_fill(coords, before), water)
        # If all rows below the surface are filled, each nonempty partner has
        # the same absolute level whenever that level intersects its chamber.
        level = int(water[:, 0].min())
        levels.append(level)
        wet = 0
        for i in ids:
            pc = vessels[i]["coords"]
            amount = int(np.count_nonzero(out[pc[:, 0], pc[:, 1]] == LIQUID))
            transfer |= len(ids) > 1 and amount != vessels[i]["volume"]
            dry += int(len(ids) > 1 and amount == 0)
            wet += int(amount > 0)
        shared |= wet >= 2
        code = vessels[ids[0]]["seal"]
        codes.append(_seal_key(code))
        histograms.append(tuple(np.bincount(code.ravel(), minlength=10)))
        if len(ids) > 1:
            assert len({_seal_key(vessels[i]["raw"]) for i in ids}) == len(ids)
    assert len(set(histograms)) == 1 and len(set(codes)) == len(groups) >= 2
    assert transfer and diagonal > 0
    assert all(_different_ablation(a, vessels, free, out, mode)
               for mode in ("independent", "global", "raw")), "parent-concept ablation failed"
    try:
        _render(a, vessels, free, "independent")
        independent_complete = True
    except ValueError:
        independent_complete = False
    norm = np.where(free, np.where(a == LIQUID, 1, 0), np.where(seals, np.where(a == ERASE, 3, 4), np.where(a == 0, 0, 2)))
    return {"shape": a.shape, "vessels": len(vessels), "circuits": len(groups),
            "group_sizes": tuple(sorted(map(len, groups))), "levels": tuple(levels),
            "wall": vessels[0]["wall"], "seal_shape": vessels[0]["seal"].shape,
            "dry": dry, "shared": shared, "diagonal_orbits": diagonal,
            "volume": int((a == LIQUID).sum()), "sources": input_components,
            "floors": floor_shapes, "norm": _seal_key(norm),
            "independent_complete": independent_complete}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _seal_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _features(pair)
            assert 3 <= f["vessels"] <= 6
            stats[split].append(f)
    assert any(f["group_sizes"] == (1, 2) and f["shared"] for f in stats["train"])
    assert any(f["group_sizes"] == (2, 2) and f["dry"] >= 2 and f["independent_complete"]
               for f in stats["train"])
    assert all(max(f["group_sizes"]) == 2 for f in stats["train"])
    for f in stats["test"]:
        assert 3 in f["group_sizes"] and f["dry"] >= 1 and f["shared"]


def _signature(puzzle):
    return tuple(tuple((_seal_key(_grid(p["input"])), _seal_key(_grid(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    """Bounded semantic, variation, determinism, API and negative checks."""
    shapes, walls, seal_shapes, floors, volumes, sources, layouts = (set() for _ in range(7))
    signatures, sizes, circuits, records = set(), set(), set(), []
    # Directly exercise the termination guard, including both taught cases
    # and the held-out three-vessel relation, under fresh seals and palettes.
    for guard_seed in range(4):
        for kind in range(4):
            pair = _make_pair(np.random.default_rng(7000 + 10 * guard_seed + kind),
                              kind, _emergency=True)
            _features(pair)
    for sh in range(3, 6):
        for sw in range(3, 6):
            code = _make_codes(np.random.default_rng(sh * 100 + sw), sh, sw, 2, 2)[0]
            used = set()
            for _ in range(4):
                raw = _distinct_witness_mask(code, used)
                assert np.array_equal(_repair(raw), code)
                used.add(_seal_key(raw))
            assert len(used) == 4
    for seed in range(42):
        nt = 2 + seed % 9
        ns = min(1 + seed % 4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ns)
        signatures.add(_signature(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            f = _features(pair)
            records.append(f)
            shapes.add(f["shape"])
            walls.add(f["wall"])
            seal_shapes.add(f["seal_shape"])
            floors.update(f["floors"])
            volumes.add(f["volume"])
            sources.add(f["sources"])
            sizes.add(f["vessels"])
            circuits.add(f["circuits"])
            layouts.add(f["norm"])
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 35 and walls == set(range(2, 9))
    assert len(seal_shapes) >= 7 and len(floors) >= 60 and len(volumes) >= 35
    assert sizes == {3, 4, 5, 6} and circuits == {2, 3} and len(sources) >= 5
    assert sum(f["dry"] > 0 for f in records) >= 80
    assert sum(f["shared"] for f in records) >= 80
    assert any(h > w for h, w in shapes) and any(w > h for h, w in shapes)
    # Every explicit count pair; the public default is tested for determinism.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    assert _signature(generate_puzzle(73)) == _signature(generate_puzzle(73))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, "1"), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(91)
    for corruption in ("water", "seal", "solid"):
        bad = {s: [{k: _grid(a) for k, a in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        vs, free, seals = _parse(pair["input"])
        eligible = pair["output"] == LIQUID if corruption == "water" else seals if corruption == "seal" else ~free & ~seals
        point = tuple(np.argwhere(eligible)[0])
        pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted a corrupt output")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42-seed study and bounded API/negative checks)")
