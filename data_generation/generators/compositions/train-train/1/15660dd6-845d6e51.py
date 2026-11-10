"""Shape-registered, two-parameter pattern transfer. Standard library + NumPy."""
from __future__ import annotations

import numpy as np


_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate(cells, turns=1):
    result = _normalize(cells)
    for _ in range(turns % 4):
        result = _normalize([(c, -r) for r, c in result])
    return result


def _canonical(cells):
    return min(_rotate(cells, k) for k in range(4))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack, cells = [(r0, c0)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                y, x = r + dr, c + dc
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        result.append(tuple(sorted(cells)))
    return result


def _tile(grid, cells, background):
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - y + 1
    w = max(c for r, c in cells) - x + 1
    tile = np.full((h, w), background, dtype=np.int8)
    for r, c in cells:
        tile[r - y, c - x] = grid[r, c]
    return tile, (y, x)


def _shape_array(shape):
    a = np.zeros((1 + max(r for r, c in shape), 1 + max(c for r, c in shape)), dtype=bool)
    for r, c in shape:
        a[r, c] = True
    return a


def _orbits(shape):
    """Cell orbits under every rotation preserving this displayed silhouette."""
    mask = _shape_array(shape)
    symmetries = [k for k in range(4) if np.array_equal(np.rot90(mask, k), mask)]
    unassigned = set(shape)
    orbits = []
    while unassigned:
        cell = min(unassigned)
        point = np.zeros(mask.shape, dtype=bool)
        point[cell] = True
        orbit = set()
        for k in symmetries:
            orbit.update(tuple(map(int, p)) for p in np.argwhere(np.rot90(point, k)))
        unassigned.difference_update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return orbits


def _analyze(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    lines = [r for r in range(1, a.shape[0] - 1)
             if int(a[r, 0]) != background and np.all(a[r] == a[r, 0])]
    _require(len(lines) == 1, "expected one full catalog divider")
    divider = lines[0]
    structural = int(a[divider, 0])
    header = _components(a[:divider] != background)
    ink_keys, prototypes = [], []
    for cells in header:
        colors = {int(a[p]) for p in cells}
        tile, _ = _tile(a, cells, background)
        if len(colors) == 1:
            prototypes.append((cells, next(iter(colors))))
        else:
            _require(tile.shape == (3, 3) and len(cells) == 9, "invalid ink key")
            base, placeholder = int(tile[0, 0]), int(tile[1, 1])
            border = np.ones((3, 3), dtype=bool)
            border[1, 1] = False
            _require(np.all(tile[border] == base) and base != placeholder, "invalid framed key")
            ink_keys.append((base, placeholder))
    _require(len(ink_keys) == 1 and 2 <= len(prototypes) <= 3, "invalid catalog")
    base, placeholder = ink_keys[0]
    roles = (background, structural, base, placeholder)
    _require(len(set(roles)) == 4, "ink roles must be distinct")
    by_shape, labels = {}, set()
    for cells, label in prototypes:
        shape = _canonical(cells)
        _require(shape not in by_shape and label not in labels and label not in roles,
                 "ambiguous catalog class")
        by_shape[shape] = {"label": label, "prototype": cells, "templates": [], "swatches": []}
        labels.add(label)
    field_mask = a != background
    field_mask[:divider + 1] = False
    components = _components(field_mask)
    for cells in components:
        shape = _canonical(cells)
        _require(shape in by_shape, "unmatched field silhouette")
        colors = {int(a[p]) for p in cells}
        tile, pos = _tile(a, cells, background)
        item = {"cells": cells, "tile": tile, "pos": pos}
        if len(colors) == 1:
            accent = next(iter(colors))
            _require(accent not in set(roles) | labels, "accent collides with another role")
            item["accent"] = accent
            by_shape[shape]["swatches"].append(item)
        else:
            _require(colors == {base, placeholder, structural}, "stencil must contain all three inks")
            by_shape[shape]["templates"].append(item)
    for shape, info in by_shape.items():
        _require(len(info["templates"]) >= 2 and len(info["swatches"]) >= 1,
                 "each class needs repeated stencils and a swatch")
        reference = info["templates"][0]["tile"]
        for stencil in info["templates"]:
            _require(any(np.array_equal(np.rot90(reference, k), stencil["tile"]) for k in range(4)),
                     "conflicting repeated stencils")
        # All silhouette-compatible rotations must agree, including symmetric shapes.
        _require(all(not np.array_equal(np.rot90(reference != background, k), reference != background)
                     or np.array_equal(np.rot90(reference, k), reference) for k in range(4)),
                 "pattern breaks silhouette symmetry")
    return a, {"background": background, "structural": structural,
               "base": base, "placeholder": placeholder, "divider": divider,
               "classes": by_shape, "roles": roles}


def _registered_tiles(reference, target, background):
    return [np.rot90(reference, k) for k in range(4)
            if np.array_equal(np.rot90(reference != background, k), target != background)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _analyze(input_grid)
    output = a.copy()
    bg, base, placeholder = info["background"], info["base"], info["placeholder"]
    for cls in info["classes"].values():
        reference = cls["templates"][0]["tile"]
        for swatch in cls["swatches"]:
            aligned = _registered_tiles(reference, swatch["tile"], bg)
            _require(bool(aligned) and all(np.array_equal(aligned[0], x) for x in aligned),
                     "ambiguous registration")
            stencil = aligned[0]
            decoded = stencil.copy()
            decoded[stencil == base] = cls["label"]
            decoded[stencil == placeholder] = swatch["accent"]
            y, x = swatch["pos"]
            region = output[y:y + decoded.shape[0], x:x + decoded.shape[1]]
            mask = stencil != bg
            region[mask] = decoded[mask]
    return output


def _random_shape(rng, area=None):
    for _ in range(300):
        n = int(rng.integers(5, 11)) if area is None else area
        cells = {(0, 0)}
        while len(cells) < n:
            frontier = {(r + dr, c + dc) for r, c in cells for dr, dc in _NEIGHBORS} - cells
            options = []
            for point in sorted(frontier):
                trial = cells | {point}
                if max(r for r, c in trial) - min(r for r, c in trial) < 4 and max(c for r, c in trial) - min(c for r, c in trial) < 4:
                    options.append(point)
            if not options:
                break
            cells.add(options[int(rng.integers(len(options)))])
        if len(cells) == n:
            shape = _canonical(cells)
            if len(_orbits(shape)) >= 3:
                return shape
    raise RuntimeError("shape sampling failed")


def _anchors(rng):
    # Equal area and bounding-box dimensions defeat area/box-only classification.
    for _ in range(100):
        first = _random_shape(rng, int(rng.integers(6, 9)))
        if len({_rotate(first, k) for k in range(4)}) < 4:
            continue
        mirror = _canonical([(r, -c) for r, c in first])
        candidates = [mirror] if mirror != first and rng.random() < 0.45 else []
        candidates.extend(_random_shape(rng, len(first)) for _ in range(20))
        box = sorted(_shape_array(first).shape)
        for second in candidates:
            if second != first and sorted(_shape_array(second).shape) == box and len({_rotate(second, k) for k in range(4)}) == 4:
                return [first, second]
    raise RuntimeError("anchor sampling failed")


def _pattern(rng, shape):
    orbits = _orbits(shape)
    assignments = rng.integers(0, 3, size=len(orbits))
    assignments[rng.permutation(len(orbits))[:3]] = rng.permutation(3)
    pattern = np.full(_shape_array(shape).shape, -1, dtype=np.int8)
    for cells, value in zip(orbits, assignments):
        for p in cells:
            pattern[p] = value
    return pattern


def _colored(pattern, colors, background, turns):
    p = np.rot90(pattern, turns)
    tile = np.full(p.shape, background, dtype=np.int8)
    for i, color in enumerate(colors):
        tile[p == i] = color
    return tile


def _pack(rng, height, width, items, y0, y1, background):
    """Random placement with an eight-neighbor moat; no fixed object slots."""
    order = sorted(range(len(items)), key=lambda i: -items[i].size)
    for _ in range(100):
        grid = np.full((height, width), background, dtype=np.int8)
        blocked = np.zeros((height, width), dtype=bool)
        placements = [None] * len(items)
        success = True
        for i in order:
            tile = items[i]
            h, w = tile.shape
            cells = np.argwhere(tile != background)
            placed = False
            for _ in range(400):
                y = int(rng.integers(y0, y1 - h + 1))
                x = int(rng.integers(1, width - w))
                ys, xs = cells[:, 0] + y, cells[:, 1] + x
                if np.any(blocked[ys, xs]):
                    continue
                grid[ys, xs] = tile[cells[:, 0], cells[:, 1]]
                for r, c in zip(ys, xs):
                    blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
                placements[i] = (y, x)
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid, placements
    raise RuntimeError("packing failed")


def _make_pair(rng, shapes, patterns, roles, labels, first=False):
    bg, structural, base, placeholder = roles
    available = [c for c in range(10) if c not in set(roles) | set(labels)]
    header_items = []
    key = np.full((3, 3), base, dtype=np.int8)
    key[1, 1] = placeholder
    header_items.append(key)
    for shape, label in zip(shapes, labels):
        mask = np.rot90(_shape_array(shape), int(rng.integers(4)))
        header_items.append(np.where(mask, label, bg).astype(np.int8))
    items, decoded, swatch_flags = [], [], []
    for index, (shape, pattern, label) in enumerate(zip(shapes, patterns, labels)):
        # Two reference copies remain the common simple case; three is legal too.
        n_templates = 2 if first else int(rng.choice([2, 2, 3]))
        n_swatches = (2 if index == 0 else 1) if first else int(rng.integers(1, 4))
        turns = [int(rng.integers(4)) for _ in range(n_templates + n_swatches)]
        if first:
            turns[1] = (turns[0] + int(rng.integers(1, 4))) % 4
        # At least one target is registered at a different orientation from reference.
        turns[n_templates] = (turns[0] + int(rng.integers(1, 4))) % 4
        if first and index == 0:
            turns[n_templates + 1] = (turns[n_templates] + int(rng.integers(1, 4))) % 4
        accents = [int(rng.choice(available)) for _ in range(n_swatches)]
        if first and index == 0:
            accents[:2] = [int(x) for x in rng.choice(available, 2, replace=False)]
        for j, turn in enumerate(turns):
            if j < n_templates:
                tile = _colored(pattern, (base, placeholder, structural), bg, turn)
                target = tile.copy()
            else:
                accent = accents[j - n_templates]
                mask = np.rot90(pattern >= 0, turn)
                tile = np.where(mask, accent, bg).astype(np.int8)
                target = _colored(pattern, (label, accent, structural), bg, turn)
            items.append(tile)
            decoded.append(target)
            swatch_flags.append(j >= n_templates)
    # Dimensions overlap across splits and depend on actual object load.
    count = len(items)
    width = int(rng.integers(23 if len(shapes) == 2 else 26, 31))
    divider = int(rng.integers(6, 8))
    field_height = int(rng.integers(16 if count <= 8 else 20, 24))
    height = min(30, divider + 1 + field_height)
    header_grid, _ = _pack(rng, height, width, header_items, 1, divider - 1, bg)
    field_grid, placements = _pack(rng, height, width, items, divider + 2, height - 1, bg)
    a = header_grid
    a[divider + 1:] = field_grid[divider + 1:]
    a[divider] = structural
    b = a.copy()
    for tile, pos, is_swatch in zip(decoded, placements, swatch_flags):
        if is_swatch:
            y, x = pos
            mask = tile != bg
            view = b[y:y + tile.shape[0], x:x + tile.shape[1]]
            view[mask] = tile[mask]
    return {"input": a, "output": b}


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def _identifier(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(int(seed))
    anchors = _anchors(rng)
    anchor_patterns = [_pattern(rng, s) for s in anchors]
    stable_inks = bool(rng.random() < 0.25)
    fixed_roles = tuple(map(int, rng.permutation(10)[:4]))
    puzzle, seen = {"train": [], "test": []}, set()
    used_train_shapes, first_label = set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            first = split == "train" and index < 2
            if first:
                shapes, patterns = list(anchors), [p.copy() for p in anchor_patterns]
            else:
                n = int(rng.integers(2, 4))
                shapes = [anchors[int(rng.integers(2))]]
                while len(shapes) < n:
                    shape = _random_shape(rng)
                    if shape not in shapes and (split != "test" or shape not in used_train_shapes):
                        shapes.append(shape)
                patterns = [_pattern(rng, s) for s in shapes]
            if split == "train":
                used_train_shapes.update(shapes)
            for _ in range(30):
                roles = fixed_roles if stable_inks else tuple(map(int, rng.permutation(10)[:4]))
                labels = list(map(int, rng.choice([c for c in range(10) if c not in roles], len(shapes), replace=False)))
                if split == "train" and index == 1 and labels[0] == first_label:
                    continue
                try:
                    pair = _make_pair(rng, shapes, patterns, roles, labels, first)
                except RuntimeError:
                    continue
                key = _identifier(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_label = labels[0]
                    break
            else:
                raise RuntimeError("distinct pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, info = _analyze(pair["input"])
            b = _grid(pair["output"])
            bg, divider = info["background"], info["divider"]
            _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
                     and np.all(a[:divider, (0, -1)] == bg)
                     and np.all(a[divider + 1:, (0, -1)] == bg), "missing canvas margin")
            _require(np.all(a[divider - 1] == bg) and np.all(a[divider + 1] == bg),
                     "object touches divider")
            object_mask = a != bg
            object_mask[divider] = False
            owners = np.zeros(a.shape, dtype=int)
            for owner, cells in enumerate(_components(object_mask), 1):
                for p in cells:
                    owners[p] = owner
            for x, y in ((owners[:-1, :-1], owners[1:, 1:]),
                         (owners[:-1, 1:], owners[1:, :-1])):
                _require(not np.any((x > 0) & (y > 0) & (x != y)), "objects touch diagonally")
            _require(a.shape == b.shape and _identifier(a) not in seen, "duplicate input or dimension change")
            seen.add(_identifier(a))
            _require(np.array_equal(solve(a), b), "incorrect output")
            allowed = np.zeros(a.shape, dtype=bool)
            ablate_a = a.copy()
            ablate_b = b.copy()
            first_label = next(iter(info["classes"].values()))["label"]
            rotated = False
            for shape, cls in info["classes"].items():
                _require(2 <= len(cls["templates"]) <= 3 and 1 <= len(cls["swatches"]) <= 3, "invalid multiplicity")
                ref = cls["templates"][0]["tile"]
                for swatch in cls["swatches"]:
                    rotated |= not np.array_equal(ref != info["background"], swatch["tile"] != info["background"])
                    for p in swatch["cells"]:
                        allowed[p] = True
                        ablate_a[p] = cls["label"]  # B's plain shape recoloring.
                        if b[p] == cls["label"]:
                            ablate_b[p] = first_label  # Ignore shape-addressed label.
                    # Direct per-cell semantic checks, including preserved structural ink.
                    aligned = _registered_tiles(ref, swatch["tile"], info["background"])[0]
                    y, x = swatch["pos"]
                    for r, c in np.argwhere(aligned != info["background"]):
                        source = int(aligned[r, c])
                        expected = cls["label"] if source == info["base"] else swatch["accent"] if source == info["placeholder"] else info["structural"]
                        _require(int(b[y + r, x + c]) == expected, "substitution/registration error")
            _require(np.array_equal(a[~allowed], b[~allowed]), "changed a non-impression cell")
            _require(not np.array_equal(ablate_a, b), "A contribution has no effect")
            _require(not np.array_equal(ablate_b, b), "B contribution has no effect")
            info["rotated"] = rotated
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    common = set(first["classes"]) & set(second["classes"])
    _require(len(common) == 2, "first demonstrations must share two classes")
    _require(any(first["classes"][s]["label"] != second["classes"][s]["label"] for s in common),
             "local catalog remapping not demonstrated")
    _require(first["rotated"] and second["rotated"], "registration not demonstrated")
    for info in (first, second):
        _require(any(len({x["accent"] for x in c["swatches"]}) >= 2 for c in info["classes"].values()),
                 "independent accent substitution not demonstrated")
    train_shapes = set().union(*(set(i["classes"]) for i in analyses["train"]))
    _require(all(set(i["classes"]) - train_shapes for i in analyses["test"]), "missing test silhouette shift")


def self_test():
    """Bounded structural, semantic, counterfactual and diversity checks on 42 seeds."""
    layouts, roles, shapes, counts, patterns, signatures = set(), set(), set(), set(), set(), set()
    symmetric = chiral_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_identifier(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, info = _analyze(pair["input"])
                layouts.add((a.shape, info["divider"]))
                roles.add(info["roles"])
                counts.add(len(info["classes"]))
                shapes.update(info["classes"])
                for shape, cls in info["classes"].items():
                    symmetric += len({_rotate(shape, k) for k in range(4)}) < 4
                    mirror = _canonical([(r, -c) for r, c in shape])
                    chiral_pairs += mirror != shape and mirror in info["classes"]
                    ref = cls["templates"][0]["tile"]
                    mapping = {info["background"]: -1, info["base"]: 0, info["placeholder"]: 1, info["structural"]: 2}
                    normalized = np.vectorize(mapping.__getitem__)(ref)
                    patterns.add(min((np.rot90(normalized, k).shape, np.rot90(normalized, k).tobytes()) for k in range(4)))
        # Change the catalog labels only: exactly that class's base-ink destinations respond.
        a, info = _analyze(puzzle["train"][0]["input"])
        classes = list(info["classes"].values())
        mutated = a.copy()
        for r, c in classes[0]["prototype"]:
            mutated[r, c] = classes[1]["label"]
        for r, c in classes[1]["prototype"]:
            mutated[r, c] = classes[0]["label"]
        altered = solve(mutated)
        original = solve(a)
        expected = original.copy()
        for cls, other in ((classes[0], classes[1]), (classes[1], classes[0])):
            for swatch in cls["swatches"]:
                for p in swatch["cells"]:
                    if original[p] == cls["label"]:
                        expected[p] = other["label"]
        # Catalog itself is retained from the modified input.
        expected[:info["divider"]] = mutated[:info["divider"]]
        assert np.array_equal(altered, expected)
        # Change one solid swatch: only placeholder destinations, plus its unchanged
        # accent-colored input cells, follow the newly supplied accent.
        accents = sorted(set(range(10)) - set(info["roles"]) - {c["label"] for c in classes})
        sw = classes[0]["swatches"][0]
        new_accent = next(c for c in accents if c != sw["accent"])
        mutated = a.copy()
        for p in sw["cells"]:
            mutated[p] = new_accent
        expected = original.copy()
        for p in sw["cells"]:
            if original[p] == sw["accent"]:
                expected[p] = new_accent
        assert np.array_equal(solve(mutated), expected)
    assert len(signatures) == 42 and len(layouts) >= 20 and len(roles) >= 30
    assert len(shapes) >= 80 and len(patterns) >= 120 and counts == {2, 3}
    assert symmetric > 0 and chiral_pairs > 0
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(10000 + n * 10 + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(784, 3, 2), generate_puzzle(784, 3, 2)
    assert all(np.array_equal(p[k], q[k]) for s in ("train", "test") for p, q in zip(a[s], b[s]) for k in ("input", "output"))
