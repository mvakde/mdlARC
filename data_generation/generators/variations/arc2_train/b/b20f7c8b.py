"""Local four-entry shape/color codebook ARC puzzles."""
from itertools import combinations
import numpy as np

SOURCE_PAIRS = {
    (18, 22, 3513855150016345321, 1479185161926953269),
    (18, 22, 7924227870349047043, 8814906939542654520),
    (18, 22, 11909819175470289107, 9266259903188234723),
    (18, 22, 6406472543003140345, 2842670027703653691),
}


def key(a):
    return tuple(int(x) for x in np.asarray(a).flat)


def variants(a):
    out = {}
    for turn in range(4):
        rotated = np.rot90(a, turn)
        for mask in (rotated, np.fliplr(rotated)):
            out[key(mask)] = mask.copy()
    return [out[k] for k in sorted(out)]


def canonical(a):
    return min(key(v) for v in variants(a))


def valid_glyph(a):
    a = np.asarray(a)
    return (a.shape == (3, 3) and np.all((a == 0) | (a == 1))
            and 3 <= int(a.sum()) <= 7 and a[0].any() and a[2].any()
            and a[:, 0].any() and a[:, 2].any())


def shape_banks():
    banks = {1: set(), 2: set(), 4: set(), 8: set()}
    for bits in range(512):
        a = np.array([(bits >> i) & 1 for i in range(9)], dtype=np.int8).reshape(3, 3)
        if valid_glyph(a):
            banks[len(variants(a))].add(canonical(a))
    return {n: tuple(sorted(s)) for n, s in banks.items()}


BANKS = shape_banks()
SHAPES = tuple(sorted(set().union(*BANKS.values())))


def mask(shape):
    return np.array(shape, dtype=np.int8).reshape(3, 3)


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def fingerprint(a):
    value = 1469598103934665603
    for cell in a.flat:
        value = ((value ^ int(cell)) * 1099511628211) & ((1 << 64) - 1)
    return value


def source_pair_copy(source, target):
    return (source.shape[0], source.shape[1], fingerprint(source),
            fingerprint(target)) in SOURCE_PAIRS


def counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} outside legal range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def roles(rng, base, forbidden=()):
    result = list(base)
    for i in range(4):
        if result[i] in forbidden and i < 2 or rng.random() < 0.32:
            choices = [c for c in range(10) if c not in result and (i >= 2 or c not in forbidden)]
            if choices:
                result[i] = int(pick(rng, choices))
    return tuple(result)


def labels(rng, palette, required=()):
    background, catalogue_bg, field, motif = palette
    choices = [c for c in range(10) if c not in (background, catalogue_bg)]
    required = set(required)
    assert required.issubset(choices) and len(required) <= 4
    groups = {0: [], 1: [], 2: []}
    for option in combinations(choices, 4):
        if required.issubset(option):
            groups[len(set(option) & {field, motif})].append(option)
    overlap = pick(rng, [n for n in groups if groups[n]])
    return tuple(int(x) for x in pick(rng, groups[overlap]))


def layout(rng, side, height, width):
    work = width - 6
    offset = 6 if side == "left" else 0
    top_max = height // 2 - 5
    bottom_min = height // 2 + 1
    left_max = work // 2 - 5
    right_min = work // 2 + 1
    positions = ((int(rng.integers(1, top_max + 1)), offset + int(rng.integers(1, left_max + 1))),
                 (int(rng.integers(1, top_max + 1)), offset + int(rng.integers(right_min, work - 5))),
                 (int(rng.integers(bottom_min, height - 5)), offset + int(rng.integers(1, left_max + 1))),
                 (int(rng.integers(bottom_min, height - 5)), offset + int(rng.integers(right_min, work - 5))))
    # Individual gaps include zero: touching slots remain separated by their distinct colors.
    for _ in range(30):
        gaps = [int(x) for x in rng.integers(0, 4, size=3)]
        if sum(gaps) <= height - 12:
            break
    else:
        gaps = [0, 0, 0]
    top = int(rng.integers(0, height - 12 - sum(gaps) + 1))
    rows = [top]
    for gap in gaps:
        rows.append(rows[-1] + 3 + gap)
    columns = [int(rng.integers(1, 3)) for _ in range(4)]
    return positions, rows, columns


def orient(a, rng, operation):
    if operation == "identity":
        return a.copy()
    if operation == "rotation":
        options = [np.rot90(a, k) for k in (1, 2, 3) if not np.array_equal(np.rot90(a, k), a)]
    elif operation == "reflection":
        rotations = {key(np.rot90(a, k)) for k in range(4)}
        options = [np.fliplr(np.rot90(a, k)) for k in range(4)
                   if key(np.fliplr(np.rot90(a, k))) not in rotations]
    else:
        options = variants(a)
    return pick(rng, options).copy() if options else a.copy()


def make_pair(rng, side, palette, mapping, solids, teaching=False):
    height, width = int(rng.integers(18, 27)), int(rng.integers(22, 31))
    background, catalogue_bg, field, motif = palette
    cards, rows, columns = layout(rng, side, height, width)
    strip = 0 if side == "left" else width - 6
    source = np.full((height, width), background, dtype=np.int8)
    source[:, strip:strip + 6] = catalogue_bg
    displayed = {}
    order = list(mapping)
    rng.shuffle(order)
    for shape, row, column in zip(order, rows, columns):
        a = pick(rng, variants(mask(shape)))
        displayed[shape] = a
        view = source[row:row + 3, strip + column:strip + column + 3]
        view[a.astype(bool)] = mapping[shape]
    target = source.copy()
    order = list(mapping)
    rng.shuffle(order)
    for (row, column), shape in zip(cards, order):
        label, a = mapping[shape], displayed[shape]
        if shape in solids:
            source[row:row + 5, column:column + 5] = label
            target[row:row + 5, column:column + 5] = field
            center = target[row + 1:row + 4, column + 1:column + 4]
            center[a.astype(bool)] = motif
        else:
            orbit = len(variants(a))
            operation = ("reflection" if orbit == 8 else "rotation" if orbit in (2, 4)
                         else "identity") if teaching else pick(rng, ("identity", "rotation", "reflection", "random"))
            b = orient(a, rng, operation)
            source[row:row + 5, column:column + 5] = field
            center = source[row + 1:row + 4, column + 1:column + 4]
            center[b.astype(bool)] = motif
            target[row:row + 5, column:column + 5] = label
    return {"input": source, "output": target}


def components(binary):
    seen = np.zeros(binary.shape, dtype=bool)
    result = []
    for row, column in np.argwhere(binary):
        if seen[row, column]:
            continue
        queue = [(int(row), int(column))]
        seen[row, column] = True
        group = []
        while queue:
            r, c = queue.pop()
            group.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1] and binary[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(group)
    return result


def parse_input(source):
    height, width = source.shape
    candidates = []
    for side, strip in (("left", 0), ("right", width - 6)):
        region = source[:, strip:strip + 6]
        catalogue_bg = int(np.bincount(region.flat, minlength=10).argmax())
        entries = {}
        occupied = np.zeros(region.shape, dtype=bool)
        for label in set(int(c) for c in region.flat) - {catalogue_bg}:
            points = np.argwhere(region == label)
            r0, c0 = points.min(axis=0)
            r1, c1 = points.max(axis=0)
            if (r1 - r0, c1 - c0) != (2, 2) or c0 not in (1, 2):
                break
            glyph = (region[r0:r0 + 3, c0:c0 + 3] == label).astype(np.int8)
            if not valid_glyph(glyph) or np.any(occupied[r0:r0 + 3, c0:c0 + 3]):
                break
            occupied[r0:r0 + 3, c0:c0 + 3] = glyph.astype(bool)
            entries[label] = (glyph, int(r0), int(c0))
        if len(entries) != 4 or len({canonical(v[0]) for v in entries.values()}) != 4:
            continue
        if np.any(region[~occupied] != catalogue_bg):
            continue
        work = source[:, 6:] if side == "left" else source[:, :width - 6]
        background = int(np.bincount(work.flat, minlength=10).argmax())
        if background == catalogue_bg or background in entries:
            continue
        positions = []
        for group in components(work != background):
            coords = np.array(group)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0, c1 - c0) != (4, 4) or len(group) != 25:
                break
            positions.append((int(r0), int(c0) + (6 if side == "left" else 0)))
        if len(positions) != 4:
            continue
        positions.sort()
        candidates.append((side, background, catalogue_bg, entries, positions))
    assert len(candidates) == 1, "input must have one catalogue/card interpretation"
    return candidates[0]


def solve_input(source):
    side, background, catalogue_bg, entries, positions = parse_input(source)
    by_shape = {canonical(v[0]): label for label, v in entries.items()}
    field = motif = None
    cards = []
    for row, column in positions:
        card = source[row:row + 5, column:column + 5]
        colors = np.unique(card)
        if len(colors) == 1:
            label = int(colors[0])
            assert label in entries
            cards.append(("solid", label, None, row, column))
            continue
        border = np.concatenate((card[0], card[4], card[1:4, 0], card[1:4, 4]))
        assert len(set(int(c) for c in border)) == 1
        this_field = int(border[0])
        inks = [int(c) for c in colors if c != this_field]
        assert len(inks) == 1
        this_motif = inks[0]
        if field is None:
            field, motif = this_field, this_motif
        assert (field, motif) == (this_field, this_motif)
        glyph = (card[1:4, 1:4] == motif).astype(np.int8)
        assert valid_glyph(glyph) and canonical(glyph) in by_shape
        cards.append(("pattern", by_shape[canonical(glyph)], glyph, row, column))
    assert field is not None and motif is not None and field != motif
    assert background not in (field, motif) and catalogue_bg not in (field, motif)
    assert len({c[1] for c in cards}) == 4 and sum(c[0] == "solid" for c in cards) <= 2
    result = source.copy()
    for kind, label, glyph, row, column in cards:
        if kind == "pattern":
            result[row:row + 5, column:column + 5] = label
        else:
            result[row:row + 5, column:column + 5] = field
            center = result[row + 1:row + 4, column + 1:column + 4]
            center[entries[label][0].astype(bool)] = motif
    return result, {"side": side, "background": background, "catalogue_bg": catalogue_bg,
                    "field": field, "motif": motif, "labels": tuple(sorted(entries)),
                    "entries": entries, "cards": cards, "positions": positions,
                    "mapping": {canonical(v[0]): label for label, v in entries.items()},
                    "rows": tuple(sorted(v[1] for v in entries.values()))}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = counts(seed, rng, num_train, num_test)
    base = tuple(int(c) for c in rng.permutation(10)[:4])
    first_shapes = tuple(pick(rng, BANKS[n]) for n in (1, 2, 4, 8))
    train_sides = ["left", "right"] + [pick(rng, ("left", "right")) for _ in range(num_train - 2)]
    rng.shuffle(train_sides)
    solid_counts = [0, int(rng.integers(1, 3))] + [int(rng.integers(0, 3)) for _ in range(num_train - 2)]
    rng.shuffle(solid_counts)
    teaching = solid_counts.index(0)
    first_palette = roles(rng, base)
    first_labels = labels(rng, first_palette)
    first_map = {shape: int(label) for shape, label in zip(first_shapes, rng.permutation(first_labels))}
    puzzle = {"train": [], "test": []}
    identities, train_shapes = set(), set()

    def append(section, side, palette, mapping, solids, teach=False):
        for _ in range(30):
            pair = make_pair(rng, side, palette, mapping, solids, teach)
            source = pair["input"]
            identity = (source.shape, source.tobytes())
            if identity not in identities and not source_pair_copy(source, pair["output"]):
                identities.add(identity)
                puzzle[section].append(pair)
                return
        raise RuntimeError("could not produce a unique pair")

    for i in range(num_train):
        if i == teaching:
            shapes, palette, mapping, solids = first_shapes, first_palette, first_map, set()
        else:
            palette = roles(rng, base)
            shared = [c for c in first_labels if c not in palette[:2]]
            required = tuple(rng.choice(shared, size=min(len(shared), int(rng.integers(1, 3))), replace=False))
            chosen_labels = labels(rng, palette, required)
            familiar = tuple(first_shapes[int(j)] for j in rng.choice(4, size=2, replace=False))
            available = [shape for shape in SHAPES if shape not in familiar]
            shapes = familiar + tuple(available[int(j)] for j in rng.choice(len(available), size=2, replace=False))
            mapping = {shape: int(label) for shape, label in zip(shapes, rng.permutation(chosen_labels))}
            solids = {shapes[int(j)] for j in rng.choice(4, size=solid_counts[i], replace=False)}
        append("train", train_sides[i], palette, mapping, solids, i == teaching)
        train_shapes.update(shapes)

    test_solids = [int(rng.integers(0, 3)) for _ in range(num_test)]
    if num_test >= 2:
        test_solids[:2] = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
        rng.shuffle(test_solids)
    test_sides = [pick(rng, ("left", "right")) for _ in range(num_test)]
    if num_test >= 2:
        test_sides[:2] = ["left", "right"]
        rng.shuffle(test_sides)
    for i in range(num_test):
        selected = list(rng.choice(4, size=3, replace=False))
        stable, remapped, donor = (first_shapes[int(j)] for j in selected)
        stable_label, remapped_label = first_map[stable], first_map[donor]
        palette = roles(rng, base, (stable_label, remapped_label))
        chosen_labels = labels(rng, palette, (stable_label, remapped_label))
        available = [shape for shape in SHAPES if shape not in train_shapes]
        fresh = tuple(available[int(j)] for j in rng.choice(len(available), size=2, replace=False))
        other = [label for label in chosen_labels if label not in (stable_label, remapped_label)]
        rng.shuffle(other)
        mapping = {stable: stable_label, remapped: remapped_label, fresh[0]: other[0], fresh[1]: other[1]}
        solids = {remapped} if test_solids[i] else set()
        if test_solids[i] == 2:
            solids.add(pick(rng, [shape for shape in mapping if shape != remapped]))
        append("test", test_sides[i], palette, mapping, solids)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert source.shape == target.shape and 18 <= source.shape[0] <= 26 and 22 <= source.shape[1] <= 30
            assert not np.array_equal(source, target)
            identity = (source.shape, source.astype(np.int8).tobytes())
            assert identity not in seen and not source_pair_copy(source, target)
            seen.add(identity)
            predicted, data = solve_input(source)
            assert np.array_equal(predicted, target)
            analyses[section].append(data)
    train, test = analyses["train"], analyses["test"]
    assert {x["side"] for x in train} == {"left", "right"}
    assert any(sum(c[0] == "solid" for c in x["cards"]) == 0 for x in train)
    assert any(any(c[0] == "solid" for c in x["cards"]) for x in train)
    associations = {(shape, label) for x in train for shape, label in x["mapping"].items()}
    train_shapes = {shape for shape, label in associations}
    assert {len(variants(mask(shape))) for shape in train_shapes} == {1, 2, 4, 8}
    for x in test:
        familiar = set(x["mapping"]) & train_shapes
        fresh = set(x["mapping"]) - train_shapes
        assert len(familiar) >= 2 and len(fresh) >= 1
        assert any((shape, label) in associations for shape, label in x["mapping"].items())
        assert any(len({label for prior_shape, label in associations if prior_shape == shape}
                       | {x["mapping"][shape]}) >= 2 for shape in familiar)
    assert any(c[0] == "pattern" and np.array_equal(c[2], x["entries"][c[1]][0])
               for x in train for c in x["cards"])
    assert any(c[0] == "pattern" and len(variants(c[2])) == 8
               and key(c[2]) not in {key(np.rot90(x["entries"][c[1]][0], k)) for k in range(4)}
               for x in train for c in x["cards"])


def self_test() -> None:
    signatures, train_counts, test_counts = set(), set(), set()
    sizes, gaps, sides, overlaps, solids, orbits, direct = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes()) for section in ("train", "test") for p in puzzle[section]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _, x = solve_input(pair["input"])
                sizes.add(pair["input"].shape)
                sides.add((section, x["side"]))
                overlaps.add(len(set(x["labels"]) & {x["field"], x["motif"]}))
                solids.add((section, sum(c[0] == "solid" for c in x["cards"])))
                gaps.update(x["rows"][i + 1] - x["rows"][i] - 3 for i in range(3))
                orbits.update(len(variants(v[0])) for v in x["entries"].values())
                direct.update(np.array_equal(c[2], x["entries"][c[1]][0]) for c in x["cards"] if c[0] == "pattern")
    assert len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 20 and gaps == {0, 1, 2, 3}
    assert sides == {(section, side) for section in ("train", "test") for side in ("left", "right")}
    assert overlaps == {0, 1, 2} and orbits == {1, 2, 4, 8} and direct == {False, True}
    assert {0, 1, 2}.issubset({n for section, n in solids if section == "test"})
    for seed, nt, ne in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 7, 3),
                         (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ne is not None:
            assert len(puzzle["test"]) == ne
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(300, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for section in ("train", "test")
               for x, y in zip(a[section], b[section]) for k in ("input", "output"))
