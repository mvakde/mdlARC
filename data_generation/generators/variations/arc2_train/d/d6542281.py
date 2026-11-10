"""Complete partially erased motifs by exact colored translation."""

import hashlib
import numpy as np


_SOURCE = {
    "7a0539173fc2eada8b6a1e56bb020a780302527d247ceb137cf307116eb25898",
    "0ec3ae5ea1bea101d3a5591a7ce5acac03f842867cccfcad26ca8924973ec377",
    "f76147ffa99113062e74e0f12f3182a9498981fcfb3f328e5adfc2eec71d202a",
    "76727f3a31231287a15cc64c406bf7dcd789f26a51bad6f7cce4e6da68863d1a",
    "38e6383ad8f482dfa77a601520cec6adbb642447646d80d9a4668cb3bb3a286b",
}


def _counts(seed, rng, train, test):
    for value, low, high in ((train, 2, 10), (test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError("invalid pair count")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + seed // 4 % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _neighbors(r, c, diagonal=True):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1):
                yield r + dr, c + dc


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        todo = [remaining.pop()]
        component = set(todo)
        while todo:
            r, c = todo.pop()
            for pos in _neighbors(r, c):
                if pos in remaining:
                    remaining.remove(pos)
                    component.add(pos)
                    todo.append(pos)
        result.append(component)
    return result


def _four_components(points):
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        todo = [remaining.pop()]
        while todo:
            r, c = todo.pop()
            for pos in ((r - 1, c), (r + 1, c),
                        (r, c - 1), (r, c + 1)):
                if pos in remaining:
                    remaining.remove(pos)
                    todo.append(pos)
    return count


def _normalize(cells):
    mr = min(r for r, c, v in cells)
    mc = min(c for r, c, v in cells)
    return tuple(sorted((r - mr, c - mc, v) for r, c, v in cells))


def _rotate(coords, rng):
    points = list(coords)
    if rng.integers(2):
        points = [(r, -c) for r, c in points]
    for _ in range(int(rng.integers(4))):
        points = [(c, -r) for r, c in points]
    mr = min(r for r, c in points)
    mc = min(c for r, c in points)
    return sorted((r - mr, c - mc) for r, c in points)


def _shape(rng):
    mode = int(rng.integers(5))
    if mode == 0:
        cells = {(0, 0), (0, 1), (1, 0), (1, 1),
                 (2, 2), (2, 3), (3, 2), (3, 3)}
        if rng.integers(2):
            cells.add((-1, int(rng.integers(2))))
        if rng.integers(2):
            cells.add((4, 2 + int(rng.integers(2))))
    elif mode == 1:
        cells = {(r, c) for r in range(3) for c in range(3)
                 if (r, c) != (1, 1)}
        if rng.integers(2):
            cells.add((3, int(rng.integers(3))))
    else:
        n = int(rng.integers(5, 11))
        cells = {(2, 2)}
        while len(cells) < n:
            choices = sorted({(rr, cc) for r, c in cells
                              for rr, cc in _neighbors(r, c, False)
                              if 0 <= rr < 5 and 0 <= cc < 6
                              and (rr, cc) not in cells})
            cells.add(choices[int(rng.integers(len(choices)))])
    return _rotate(cells, rng)


def _embeddings(small, large):
    lookup = {(r, c): v for r, c, v in large}
    ar, ac, av = small[0]
    offsets = set()
    for r, c, v in large:
        if v == av:
            dr, dc = r - ar, c - ac
            if all(lookup.get((sr + dr, sc + dc)) == sv
                   for sr, sc, sv in small):
                offsets.add((dr, dc))
    return offsets


def _template(rng, coords, palette):
    if rng.integers(2):
        axis = int(rng.integers(2))
        order = sorted(coords, key=lambda p: p[axis])
        cut = int(rng.integers(2, len(coords) - 1))
        colors = {pos: palette[0 if i < cut else 1]
                  for i, pos in enumerate(order)}
        if len(palette) > 2:
            for pos in coords:
                if rng.random() < 0.22:
                    colors[pos] = palette[int(rng.integers(2, len(palette)))]
    else:
        colors = {pos: palette[int(rng.integers(len(palette)))]
                  for pos in coords}
        if len(set(colors.values())) < 2:
            colors[coords[0]] = palette[0]
            colors[coords[-1]] = palette[1]
    return tuple(sorted((r, c, colors[(r, c)]) for r, c in coords))


def _clue(rng, typ, templates, mode):
    template = templates[typ]
    n = len(template)
    options = {"single": [1, 2, 3, 4, n - 1],
               "sparse": [2, 3, 4, 5, n - 1],
               "middle": [max(2, n // 2), n - 2, n - 1],
               "near": [n - 1, n - 2, n - 3]}[mode]
    for size in options:
        size = min(n - 1, size)
        for _ in range(65):
            indices = rng.choice(n, size=size, replace=False)
            clue = tuple(template[int(i)] for i in indices)
            matches = {(i, dr, dc) for i, t in enumerate(templates)
                       for dr, dc in _embeddings(clue, t)}
            if matches == {(typ, 0, 0)}:
                return clue
    return None


def _box(origin, template):
    r, c = origin
    return (r, c, r + max(x for x, y, v in template) + 1,
            c + max(y for x, y, v in template) + 1)


def _apart(a, b):
    return (a[2] + 1 <= b[0] or b[2] + 1 <= a[0]
            or a[3] + 1 <= b[1] or b[3] + 1 <= a[1])


def _place(rng, templates, instances, height, width):
    placed = []
    boxes = []
    for index in rng.permutation(len(instances)):
        typ, partial, clue = instances[int(index)]
        template = templates[typ]
        th, tw = _box((0, 0), template)[2:]
        options = [(r, c) for r in range(1, height - th)
                   for c in range(1, width - tw)
                   if all(_apart(_box((r, c), template), box)
                          for box in boxes)]
        if not options:
            return None
        origin = options[int(rng.integers(len(options)))]
        placed.append((typ, partial, clue, origin))
        boxes.append(_box(origin, template))
    return placed


def _input_templates(grid, bg):
    components = _components(set(zip(*np.nonzero(grid != bg))))
    signatures = [_normalize([(r, c, int(grid[r, c])) for r, c in comp])
                  for comp in components]
    templates, origins = [], []
    for i, sig in enumerate(signatures):
        if len(sig) < 5 or any(len(other) > len(sig)
                              and _embeddings(sig, other)
                              for other in signatures):
            continue
        if sig not in templates:
            templates.append(sig)
            origins.append((min(r for r, c in components[i]),
                            min(c for r, c in components[i])))
    return templates, origins


def _restore_from_input(grid):
    """Infer maximal full components, then match each clue by colored offsets."""
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    templates, origins = _input_templates(grid, bg)
    if not 1 <= len(templates) <= 3:
        return None
    fixed_boxes = [_box(origin, t) for origin, t in zip(origins, templates)]
    fixed = {(r + r0, c + c0) for (r0, c0), t in zip(origins, templates)
             for r, c, v in t}
    clues = {(int(r), int(c)): int(grid[r, c])
             for r, c in zip(*np.nonzero(grid != bg))
             if (int(r), int(c)) not in fixed}
    if not clues:
        return None
    keys = {(i, r - tr, c - tc) for (r, c), color in clues.items()
            for i, t in enumerate(templates) for tr, tc, v in t
            if v == color}
    candidates = []
    for i, r0, c0 in keys:
        t = templates[i]
        box = _box((r0, c0), t)
        if (r0 < 0 or c0 < 0 or box[2] > grid.shape[0]
                or box[3] > grid.shape[1]
                or any(not _apart(box, old) for old in fixed_boxes)):
            continue
        mapping = {(r0 + r, c0 + c): v for r, c, v in t}
        inside = {(r, c): v for (r, c), v in clues.items()
                  if r0 <= r < box[2] and c0 <= c < box[3]}
        if inside and all(mapping.get(pos) == v for pos, v in inside.items()):
            candidates.append((frozenset(inside), box, mapping))
    chosen = []
    remaining = set(clues)
    while remaining:
        viable = [(covered, box, mapping)
                  for covered, box, mapping in candidates
                  if covered <= remaining
                  and all(_apart(box, old[1]) for old in chosen)]
        by_cell = {}
        for candidate in viable:
            for pos in candidate[0]:
                by_cell.setdefault(pos, []).append(candidate)
        proposals = []
        for pos in remaining:
            matches = by_cell.get(pos, [])
            if not matches:
                return None
            score = max(len(item[0]) for item in matches)
            winners = [item for item in matches if len(item[0]) == score]
            if len(winners) != 1:
                return None
            proposals.append(winners[0])
        selected = max(proposals, key=lambda item: len(item[0]))
        chosen.append(selected)
        remaining -= selected[0]
    output = grid.copy()
    for covered, box, mapping in chosen:
        for (r, c), color in mapping.items():
            if output[r, c] != bg and output[r, c] != color:
                return None
            output[r, c] = color
    # A disconnected clue may be partitioned among several apparent copies.
    # Reject a scene if an equally concise (or shorter) cover changes cells.
    visited = [0]

    def alternative(left, selected):
        visited[0] += 1
        if visited[0] > 10000:
            return True
        if not left:
            other = grid.copy()
            for support, bounds, mapping in selected:
                for (r, c), color in mapping.items():
                    other[r, c] = color
            return not np.array_equal(other, output)
        if len(selected) >= len(chosen):
            return False
        choices = None
        for pos in left:
            available = [candidate for candidate in candidates
                         if pos in candidate[0] and candidate[0] <= left
                         and all(_apart(candidate[1], item[1])
                                 for item in selected)]
            if not available:
                return False
            if choices is None or len(available) < len(choices):
                choices = available
        for candidate in sorted(choices, key=lambda x: -len(x[0])):
            if alternative(left - candidate[0], selected + [candidate]):
                return True
        return False

    if alternative(set(clues), []):
        return None
    return output


def _make_pair(rng, regime):
    train = regime == "source_train"
    ntypes = int(rng.integers(1, 3 if train else 4))
    extra = int(rng.integers(0, 2 if train else 3))
    partial_types = list(range(ntypes)) + [int(rng.integers(ntypes))
                                           for _ in range(extra)]
    nitems = ntypes + len(partial_types)
    low = 12 if nitems <= 3 else 18 if nitems <= 5 else 25
    high = 23 if train else 30
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    if nitems >= 5:
        height = max(height, 23)
        width = max(width, 23)
    bg = int(rng.integers(10))
    foreground = [v for v in range(10) if v != bg]
    rng.shuffle(foreground)
    shared = ntypes > 1 and rng.random() < 0.5
    common_palette = foreground[:int(rng.integers(2, 4))]
    templates = []
    for type_index in range(ntypes):
        if type_index == 1 and rng.random() < 0.36:
            base = templates[0]
            colors = sorted({v for r, c, v in base})
            if len(colors) >= 2:
                chosen = set(int(x) for x in rng.choice(
                    len(base), size=int(rng.integers(1, 3)), replace=False))
                altered = []
                for index, (r, c, v) in enumerate(base):
                    if index in chosen:
                        alternatives = [color for color in colors if color != v]
                        v = alternatives[int(rng.integers(len(alternatives)))]
                    altered.append((r, c, v))
                if len({v for r, c, v in altered}) >= 2:
                    templates.append(tuple(altered))
                    continue
        for attempt in range(60):
            coords = _shape(rng)
            palette = common_palette if shared else list(
                rng.choice(foreground, size=int(rng.integers(2, 5)),
                           replace=False))
            t = _template(rng, coords, palette)
            if all(not _embeddings(t, other)
                   and not _embeddings(other, t) for other in templates):
                templates.append(t)
                break
        else:
            return None
    instances = [(i, False, t) for i, t in enumerate(templates)]
    modes = ("single", "sparse", "middle", "near")
    for typ in partial_types:
        clue = _clue(rng, typ, templates, modes[int(rng.integers(4))])
        if clue is None:
            return None
        instances.append((typ, True, clue))
    placements = _place(rng, templates, instances, height, width)
    if placements is None:
        return None
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for typ, partial, clue, (r0, c0) in placements:
        for r, c, v in templates[typ]:
            out[r0 + r, c0 + c] = v
        for r, c, v in clue:
            inp[r0 + r, c0 + c] = v
    inferred = _restore_from_input(inp)
    if inferred is None or not np.array_equal(inferred, out):
        return None
    return {"input": inp, "output": out}


def _signature(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(int(seed), rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", num_train), ("test", num_test)):
        for _ in range(number):
            for attempt in range(200):
                pair = _make_pair(rng, "source_train" if split == "train"
                                  else "target_test")
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a recoverable pair")
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            ikey, okey = _signature(inp), _signature(out)
            assert ikey not in inputs and (ikey, okey) not in pairs
            inputs.add(ikey)
            pairs.add((ikey, okey))
            digest = hashlib.sha256(bytes(inp.shape) +
                                    inp.astype(np.int8).tobytes() +
                                    out.astype(np.int8).tobytes()).hexdigest()
            assert digest not in _SOURCE
            colors, counts = np.unique(inp, return_counts=True)
            bg = int(colors[int(np.argmax(counts))])
            changed = inp != out
            assert np.any(changed) and np.all(inp[changed] == bg)
            assert np.all(out[changed] != bg)
            inferred = _restore_from_input(inp)
            assert inferred is not None and np.array_equal(inferred, out)
            templates, origins = _input_templates(inp, bg)
            assert 1 <= len(templates) <= (2 if split == "train" else 3)
            assert len(templates) == len(origins)
            components = _components(set(zip(*np.nonzero(out != bg))))
            assert len(components) > len(templates)
            groups = {}
            for component in components:
                sig = _normalize([(r, c, int(out[r, c]))
                                  for r, c in component])
                assert sig in templates
                groups[sig] = groups.get(sig, 0) + 1
            assert all(groups.get(t, 0) >= 2 for t in templates)
            assert len(components) <= (5 if split == "train" else 8)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p["train"][0]["input"]) for p in puzzles}) >= 40
    shapes, backgrounds, structures, motifs = set(), set(), set(), set()
    two_color = near = sparse = mixed = shared = 0
    no_beacon = disconnected = singleton = similar = 0
    small = large = lesser_test = lesser_count = mixed_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        connectivity_modes = set()
        train_areas = [p["input"].size for p in puzzle["train"]]
        train_occurrences = []
        for pair in puzzle["train"]:
            inp, out = pair["input"], pair["output"]
            colors, counts = np.unique(inp, return_counts=True)
            bg = int(colors[int(np.argmax(counts))])
            train_occurrences.append(len(_components(set(zip(*np.nonzero(out != bg))))))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                colors, counts = np.unique(inp, return_counts=True)
                bg = int(colors[int(np.argmax(counts))])
                templates, origins = _input_templates(inp, bg)
                shapes.add(inp.shape)
                backgrounds.add(bg)
                components = _components(set(zip(*np.nonzero(out != bg))))
                structures.add((inp.shape, tuple(sorted(map(len, components)))))
                motifs.update(templates)
                small += int(max(inp.shape) <= 13)
                large += int(30 in inp.shape)
                if split == "test":
                    lesser_test += int(inp.size < max(train_areas))
                    lesser_count += int(len(components) < max(train_occurrences))
                two_color += sum(len({v for r, c, v in t}) == 2 for t in templates)
                if len(templates) > 1:
                    shared += int(bool({v for r, c, v in templates[0]}
                                       & {v for r, c, v in templates[1]}))
                topologies = {_four_components({(r, c) for r, c, v in t}) > 1
                              for t in templates}
                connectivity_modes.update(topologies)
                mixed += int(len(topologies) > 1)
                for i, left in enumerate(templates):
                    for right in templates[i + 1:]:
                        if ({(r, c) for r, c, v in left}
                                == {(r, c) for r, c, v in right}
                                and sum(a[2] != b[2] for a, b in zip(left, right)) <= 2):
                            similar += 1
                color_counts = {}
                for template in templates:
                    for r, c, v in template:
                        color_counts[v] = color_counts.get(v, 0) + 1
                for component in components:
                    kept = sum(inp[r, c] != bg for r, c in component)
                    near += int(kept == len(component) - 1)
                    sparse += int(kept <= 2 and kept < len(component))
                    if kept < len(component):
                        singleton += int(kept == 1)
                        clues = {(r, c) for r, c in component if inp[r, c] != bg}
                        disconnected += int(len(_components(clues)) > 1)
                        no_beacon += int(all(color_counts[int(inp[r, c])] > 1
                                             for r, c in clues))
        mixed_puzzles += int(len(connectivity_modes) > 1)
    assert len(shapes) >= 20 and len(backgrounds) >= 8
    assert len(structures) >= 80 and len(motifs) >= 80
    assert two_color >= 10 and near >= 5 and sparse >= 5 and shared >= 10
    assert no_beacon >= 100 and disconnected >= 50 and singleton >= 5
    assert mixed >= 10 and mixed_puzzles >= 15 and similar >= 5
    assert small >= 1 and large >= 5 and lesser_test >= 5 and lesser_count >= 5
    for seed, tr, te in ((7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
                         (7004, None, 1), (7005, None, 4),
                         (7006, 2, None), (7007, 10, None)):
        a, b = generate_puzzle(seed, tr, te), generate_puzzle(seed, tr, te)
        assert all(np.array_equal(x["input"], y["input"])
                   and np.array_equal(x["output"], y["output"])
                   for split in ("train", "test")
                   for x, y in zip(a[split], b[split]))
    for tr, te in ((1, None), (11, None), (None, 0),
                   (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9001, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
