"""Rank-paired paint bridges. Only standard library and NumPy are used.

All semantics are input-visible or family-fixed. Generation regimes select
distributions, never conventions required by solve().
"""

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _tile(size, color, background, kind):
    tile = np.full((size, size), color, dtype=np.int8)
    if kind == "H":
        tile[1:-1, 1:-1] = background
    return tile


def _parse(value):
    grid = _grid(value)
    background = int(grid[0, 0])
    remaining = set(map(tuple, np.argwhere(grid != background)))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for rc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if rc in remaining:
                    remaining.remove(rc)
                    stack.append(rc)
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
        size = r1-r0+1
        assert size >= 3 and c1-c0+1 == size, "non-square stamp"
        patch = grid[r0:r1+1, c0:c1+1]
        color = int(patch[0, 0])
        assert color != background
        kind = "S" if np.all(patch == color) else "H"
        assert np.array_equal(patch, _tile(size, color, background, kind)), "invalid stamp"
        # Test the entire footprint, not just its colored component.
        halo = grid[max(0, r0-1):min(grid.shape[0], r1+2),
                    max(0, c0-1):min(grid.shape[1], c1+2)].copy()
        rr, cc = r0-max(0, r0-1), c0-max(0, c0-1)
        halo[rr:rr+size, cc:cc+size] = background
        assert np.all(halo == background), "stamps must be separated"
        objects.append(dict(top=r0, left=c0, size=size, color=color, kind=kind))
    assert objects and len({o["size"] for o in objects}) == 1, "mixed stamp sizes"
    objects.sort(key=lambda o: (o["top"], o["left"]))
    queues = [[o for o in objects if o["kind"] == k] for k in ("H", "S")]
    for queue in queues:
        assert queue, "both shapes are required"
        assert len({o["top"] for o in queue}) == len(queue), "same-class top tie"
    return dict(grid=grid, background=background, size=objects[0]["size"],
                objects=objects, queues=queues)


def _pairings(a, mode="rank"):
    hollows, solids = a["queues"]
    if mode in ("horizontal", "horizontal_reverse"):
        reverse = mode.endswith("reverse")
        hollows = sorted(hollows, key=lambda o: (o["left"], o["top"]), reverse=reverse)
        solids = sorted(solids, key=lambda o: (o["left"], o["top"]), reverse=reverse)
    elif mode == "vertical_reverse":
        hollows, solids = hollows[::-1], solids[::-1]
    elif mode == "greedy":
        slots = [{}]
        for obj in a["objects"]:
            if obj["kind"] in slots[-1]:
                slots.append({})
            slots[-1][obj["kind"]] = obj
        return [(slot["H"], slot["S"]) for slot in slots if len(slot) == 2]
    elif mode == "nearest":
        free = list(solids)
        pairs = []
        for hollow in hollows:
            if not free:
                break
            solid = min(free, key=lambda o: (abs(o["top"]-hollow["top"])
                        + abs(o["left"]-hollow["left"]), o["top"], o["left"]))
            free.remove(solid)
            pairs.append((hollow, solid))
        return pairs
    pairs = list(zip(hollows, solids))
    if mode == "extend_tail":
        n = len(pairs)
        pairs += [(h, solids[-1]) for h in hollows[n:]]
        pairs += [(hollows[-1], s) for s in solids[n:]]
    return pairs


def _layers(a, mode="rank"):
    size = a["size"]
    return [dict(hollow=h, solid=s,
                 bounds=(min(h["top"], s["top"]), min(h["left"], s["left"]),
                         max(h["top"], s["top"])+size-1,
                         max(h["left"], s["left"])+size-1))
            for h, s in _pairings(a, mode)]


def _compose(a, mode="rank", reverse_priority=False, transparent=False, max_pairs=None):
    """Whole-box writer for generation and finite shortcut diagnostics."""
    result = a["grid"].copy()
    layers = _layers(a, mode)
    if max_pairs is not None:
        layers = layers[:max_pairs]
    if reverse_priority:
        layers.reverse()
    size, bg = a["size"], a["background"]
    for layer in reversed(layers):
        r0, c0, r1, c1 = layer["bounds"]
        h, s = layer["hollow"], layer["solid"]
        r, c = h["top"], h["left"]
        region = result[r0:r1+1, c0:c1+1]
        if transparent:
            mask = np.ones(region.shape, dtype=bool)
            mask[r-r0+1:r-r0+size-1, c-c0+1:c-c0+size-1] = False
            region[mask] = s["color"]
            patch = result[r:r+size, c:c+size]
            patch[0, :] = patch[-1, :] = h["color"]
            patch[:, 0] = patch[:, -1] = h["color"]
        else:
            region[:] = s["color"]
            result[r:r+size, c:c+size] = _tile(size, h["color"], bg, "H")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input, without generation state."""
    a = _parse(input_grid)
    result = a["grid"].copy()
    layers, size = _layers(a), a["size"]
    # Unlike the label writer, select the first covering layer at each pixel.
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            for layer in layers:
                r0, c0, r1, c1 = layer["bounds"]
                if r0 <= r <= r1 and c0 <= c <= c1:
                    hollow = layer["hollow"]
                    hr, hc = hollow["top"], hollow["left"]
                    if hr <= r < hr+size and hc <= c < hc+size:
                        result[r, c] = a["grid"][r, c]
                    else:
                        result[r, c] = layer["solid"]["color"]
                    break
    return result


def _witnesses(a):
    out = _compose(a)
    names = ("greedy", "horizontal", "horizontal_reverse", "vertical_reverse",
             "nearest", "extend_tail")
    found = {mode for mode in names if not np.array_equal(out, _compose(a, mode))}
    if not np.array_equal(out, _compose(a, reverse_priority=True)):
        found.add("priority")
    if not np.array_equal(out, _compose(a, transparent=True)):
        found.add("opaque_hole")
    if not np.array_equal(out, _compose(a, max_pairs=2)):
        found.add("novel_depth")
    foreground = a["grid"] != a["background"]
    if np.any(out[foreground] != a["grid"][foreground]):
        found.add("occlusion")
    if np.any(out[~foreground] != a["background"]):
        found.add("completion")
    n = min(map(len, a["queues"]))
    union = np.zeros(out.shape, dtype=bool)
    for layer in _layers(a):
        r0, c0, r1, c1 = layer["bounds"]
        union[r0:r1+1, c0:c1+1] = True
    for queue in a["queues"]:
        for obj in queue[n:]:
            r, c, s = obj["top"], obj["left"], obj["size"]
            if not union[r:r+s, c:c+s].any():
                found.add("inactive_"+obj["kind"])
    return found


def _dimensions(rng, size, count):
    # A constructive staggered packing is possible for every chosen canvas.
    options = []
    for height in range(max(8, size+count-1), 31):
        for width in range(max(8, size+2), 31):
            capacity = (width+1)//(size+1)
            bands = (count+capacity-1)//capacity
            if bands*size+count-1 <= height:
                options.append((height, width))
    assert options
    mode = int(rng.integers(4))
    if mode == 0:
        small = min(h*w for h, w in options)
        options = [(h, w) for h, w in options if h*w <= small*1.45]
    elif mode == 1:
        options = [(h, w) for h, w in options if w >= h] or options
    elif mode == 2:
        options = [(h, w) for h, w in options if h >= w] or options
    return options[int(rng.integers(len(options)))]


def _positions(rng, size, count, shape):
    height, width = shape
    rr, cc = np.indices((height-size+1, width-size+1))

    def sites(others):
        legal = (rr != 0) | (cc != 0)  # top-left encodes background
        for r, c in others:
            legal &= ((abs(rr-r) > size) | (abs(cc-c) > size)) & (rr != r)
        return np.argwhere(legal)

    positions = []
    for _ in range(6):
        positions = []
        for _ in range(count):
            choices = sites(positions)
            if not len(choices):
                break
            positions.append(tuple(map(int, choices[int(rng.integers(len(choices)))])))
        if len(positions) == count:
            break
    if len(positions) != count:
        capacity = (width+1)//(size+1)
        feasible = [b for b in range(1, count+1)
                    if b*capacity >= count and b*size+count-1 <= height]
        bands = int(rng.choice(feasible))
        amounts = [1]*bands
        for _ in range(count-bands):
            eligible = [j for j, n in enumerate(amounts) if n < capacity]
            amounts[int(rng.choice(eligible))] += 1
        vertical = rng.multinomial(height-(bands*size+count-1),
                                  np.full(bands+1, 1/(bands+1)))
        top, positions = int(vertical[0]), []
        for band, amount in enumerate(amounts):
            horizontal = rng.multinomial(width-(amount*(size+1)-1),
                                        np.full(amount+1, 1/(amount+1)))
            offsets = list(map(int, rng.permutation(amount)))
            # If both leading gaps vanish, put row zero in a nonzero column.
            if top == 0 and horizontal[0] == 0 and offsets[0] == 0 and amount > 1:
                offsets[0], offsets[1] = offsets[1], offsets[0]
            left = int(horizontal[0])
            for j in range(amount):
                c = left + int(top == 0 and amount == 1 and left == 0)
                positions.append((top+offsets[j], c))
                left += size+1+int(horizontal[j+1])
            top += size+amount+int(vertical[band+1])
    # Relocation is free over all legal coordinates, not limited to shelves.
    for _ in range(2):
        for i in rng.permutation(count):
            others = [p for j, p in enumerate(positions) if j != i]
            choices = sites(others)
            positions[int(i)] = tuple(map(int, choices[int(rng.integers(len(choices)))]))
    return sorted(positions)


def _candidate(rng, size, counts, demo_color=None, demo_kind=None, witness=False):
    nh, ns = counts
    shape = _dimensions(rng, size, nh+ns)
    positions = _positions(rng, size, nh+ns, shape)
    if witness:
        kinds = ["S", "S", "H", "H", "H" if nh > ns else "S"]
    elif demo_kind is not None:
        # The final unmatched stamp is visibly beyond every active box.
        tail = "H" if nh > ns else "S"
        kinds = list(rng.permutation(["H"]*(nh-(tail == "H"))
                                     + ["S"]*(ns-(tail == "S")))) + [tail]
    else:
        kinds = list(rng.permutation(["H"]*nh + ["S"]*ns))
    available = [c for c in range(10) if c != demo_color] if demo_color is not None else list(range(10))
    bg = int(rng.choice(available))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    if nh+ns <= 9 and rng.random() < .4:
        colors = palette[:nh+ns]
    else:
        pool = palette[:int(rng.integers(2, min(7, nh+ns)+1))]
        colors = list(map(int, rng.choice(pool, nh+ns)))
    if demo_kind is not None:
        colors[kinds.index(demo_kind)] = demo_color
    if witness:
        solid_indices = [i for i, k in enumerate(kinds) if k == "S"]
        hollow_index = kinds.index("H")
        # Paint contrast makes both depth and hole clearing visible.
        colors[solid_indices[0]] = int(rng.choice([c for c in palette if c != colors[hollow_index]]))
        colors[solid_indices[1]] = int(rng.choice([c for c in palette if c != colors[solid_indices[0]]]))
        if demo_kind == "S":
            colors[solid_indices[0]] = demo_color
            colors[solid_indices[1]] = int(rng.choice([c for c in palette if c != demo_color]))
    grid = np.full(shape, bg, dtype=np.int8)
    for (r, c), kind, color in zip(positions, kinds, colors):
        grid[r:r+size, c:c+size] = _tile(size, color, bg, kind)
    a = _parse(grid)
    return dict(input=grid, output=_compose(a)), a


def _teaching_fallback(rng, size, counts, demo_color, demo_kind):
    """Guaranteed witnesses after bounded free-placement rejection.

    Only this last-resort geometry is prescribed. It has an upper opaque
    hollow hole over the lower bridge and a lower solid inside the upper box.
    Horizontal reflection and margins preserve every semantic witness.
    """
    bg = int(rng.choice([c for c in range(10) if c != demo_color]))
    palette = list(map(int, rng.permutation([c for c in range(10)
                                            if c not in (bg, demo_color)])))
    positions = [(0, size+1), (size+3, 0), (2*size+4, 0),
                 (2*size+5, 2*size+1), (3*size+6, 0)]
    kinds = ["S", "S", "H", "H", "H" if counts[0] > counts[1] else "S"]
    colors = [palette[0], palette[1], palette[2], palette[3], palette[4]]
    colors[kinds.index(demo_kind)] = demo_color
    height, width = 4*size+6, 3*size+1
    if rng.integers(2):
        positions = [(r, width-size-c) for r, c in positions]
    # Reflection cannot move the sole top-edge tile into the top-left corner.
    dr, dc = int(rng.integers(4)), int(rng.integers(4))
    grid = np.full((height+dr, width+dc), bg, dtype=np.int8)
    for (r, c), kind, color in zip(positions, kinds, colors):
        grid[r+dr:r+dr+size, c+dc:c+dc+size] = _tile(size, color, bg, kind)
    a = _parse(grid)
    return dict(input=grid, output=_compose(a)), a


def _distinct_palette_fallback(pair, seen):
    """Ninety valid recolorings exceed the fourteen-input puzzle limit."""
    original = pair["input"]
    mask = original != original[0, 0]
    for background in range(10):
        for color in range(10):
            if color == background:
                continue
            grid = np.full(original.shape, background, dtype=np.int8)
            grid[mask] = color
            if _key(grid) not in seen:
                a = _parse(grid)
                return dict(input=grid, output=_compose(a))
    raise AssertionError("ninety recolorings cannot all be among fourteen inputs")


def _ordinary_fallback(size, counts, seen):
    """Disjoint vertical bridges and visible tails guarantee the test shift."""
    matched, tail = min(counts), abs(counts[0]-counts[1])
    extra = "H" if counts[0] > counts[1] else "S"
    stamps = []
    for i in range(matched):
        stamps += [(i+1, i*(size+1)+1, "H"),
                   (size+2+i, i*(size+1)+1, "S")]
    for i in range(tail):
        stamps.append((2*size+matched+2+i, i*(size+1)+1, extra))
    height = max(r for r, _, _ in stamps)+size+1
    width = max(c for _, c, _ in stamps)+size+1
    grid = np.zeros((height, width), dtype=np.int8)
    for r, c, kind in stamps:
        grid[r:r+size, c:c+size] = _tile(size, 1, 0, kind)
    return _distinct_palette_fallback(dict(input=grid), seen)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["tail", "depth", "scale"]))
    first_extra = str(rng.choice(["H", "S"]))
    demo_color = int(rng.integers(10))
    puzzle, seen = {"train": [], "test": []}, set()
    core = {"greedy", "horizontal", "horizontal_reverse", "vertical_reverse",
            "nearest", "priority", "opaque_hole", "occlusion", "completion"}
    for index in range(num_train+num_test):
        training = index < num_train
        if index < 2:
            extra = first_extra if index == 0 else ("S" if first_extra == "H" else "H")
            counts = (3, 2) if extra == "H" else (2, 3)
            size = int(rng.integers(3, 5))
            demo_kind = "H" if index == 0 else "S"
            required = {"extend_tail", "inactive_"+extra}
            if index == 0:
                required |= core
        else:
            demo_kind, required = None, set()
            if training:
                small = int(rng.integers(1, 3 if regime == "depth" else 4))
                gap = int(rng.integers(2))
                size = int(rng.integers(3, 5 if regime == "scale" else 6))
            elif regime == "tail":
                small, gap = int(rng.integers(1, 4)), int(rng.integers(2, 4))
                size = int(rng.integers(3, 6))
            elif regime == "depth":
                small, gap = int(rng.integers(3, 5)), int(rng.integers(2))
                size = int(rng.integers(3, 5))
            else:
                small, gap = int(rng.integers(1, 4)), int(rng.integers(2))
                size = int(rng.integers(5, 7))
            counts = (small, small+gap) if rng.integers(2) else (small+gap, small)
            if not training and regime == "tail":
                required = {"extend_tail", "inactive_"+("H" if counts[0] > counts[1] else "S")}
            elif not training and regime == "depth":
                required = {"novel_depth"}
        # A fixed, finite budget. Resampling changes only distribution controls.
        for _ in range(320):
            pair, analysis = _candidate(rng, size, counts,
                demo_color if demo_kind is not None else None,
                demo_kind, witness=(index == 0))
            if _key(pair["input"]) in seen:
                continue
            if required and not required <= _witnesses(analysis):
                continue
            break
        else:
            if index >= 2:
                pair = _ordinary_fallback(size, counts, seen)
                assert required <= _witnesses(_parse(pair["input"]))
            else:
                pair, analysis = _teaching_fallback(rng, size, counts, demo_color, demo_kind)
                assert required <= _witnesses(analysis)
                assert _key(pair["input"]) not in seen
        seen.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _parse(inp)
            assert not np.array_equal(inp, out), "vacuous pair"
            assert np.array_equal(out, _compose(a)), "incorrect compositing"
            assert np.array_equal(out, solve(inp)), "input-only prediction disagrees"
            union = np.zeros(inp.shape, dtype=bool)
            for layer in _layers(a):
                r0, c0, r1, c1 = layer["bounds"]
                union[r0:r1+1, c0:c1+1] = True
            assert np.array_equal(inp[~union], out[~union]), "changed inactive region"
            (training if split == "train" else tests).append(a)
    by_pair = [_witnesses(a) for a in training]
    witnesses = set().union(*by_pair)
    assert {"greedy", "horizontal", "horizontal_reverse", "vertical_reverse",
            "nearest", "priority", "opaque_hole", "occlusion", "completion",
            "extend_tail", "inactive_H", "inactive_S"} <= witnesses, "missing teaching evidence"
    for kind in ("H", "S"):
        assert any({"extend_tail", "inactive_"+kind} <= w for w in by_pair), "untaught inactive tail"
        assert any(abs(len(a["queues"][0])-len(a["queues"][1])) == 1 and
                   len(a["queues"][0 if kind == "H" else 1]) > len(a["queues"][1 if kind == "H" else 0])
                   for a in training), "missing tail direction"
    color_roles = {}
    for a in training:
        for o in a["objects"]:
            color_roles.setdefault(o["color"], set()).add(o["kind"])
    assert any(len(roles) == 2 for roles in color_roles.values()), "missing color-role switch"
    # Infer a shift signature from the arrays, not a stored regime.
    tail_shift = max(abs(len(a["queues"][0])-len(a["queues"][1])) for a in training) <= 1 and all(
        abs(len(a["queues"][0])-len(a["queues"][1])) >= 2 and
        {"extend_tail", "inactive_"+("H" if len(a["queues"][0]) > len(a["queues"][1]) else "S")}
        <= _witnesses(a) for a in tests)
    depth_shift = max(min(map(len, a["queues"])) for a in training) <= 2 and all(
        min(map(len, a["queues"])) >= 3 and "novel_depth" in _witnesses(a) for a in tests)
    scale_shift = max(a["size"] for a in training) <= 4 and all(a["size"] >= 5 for a in tests)
    assert tail_shift or depth_shift or scale_shift, "missing meaningful train/test shift"


def self_test():
    """Bounded 42-seed semantic and variation sweep, plus focused controls."""
    signatures, dimensions, layouts, backgrounds, sizes, counts = set(), set(), set(), set(), set(), set()
    rectangle_sizes, directions, overlap_counts = set(), set(), set()
    repeat_pairs = edge_pairs = 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _parse(pair["input"])
                dimensions.add(pair["input"].shape)
                sizes.add(a["size"])
                backgrounds.add(a["background"])
                counts.add(tuple(map(len, a["queues"])))
                layouts.add((a["size"], tuple((o["top"], o["left"], o["kind"]) for o in a["objects"])))
                repeat_pairs += len({o["color"] for o in a["objects"]}) < len(a["objects"])
                edge_pairs += any(o["top"] == 0 or o["left"] == 0 or
                                  o["top"]+a["size"] == a["grid"].shape[0] or
                                  o["left"]+a["size"] == a["grid"].shape[1] for o in a["objects"])
                layers = _layers(a)
                union = np.zeros(a["grid"].shape, dtype=int)
                for layer in layers:
                    r0, c0, r1, c1 = layer["bounds"]
                    rectangle_sizes.add((r1-r0+1, c1-c0+1))
                    union[r0:r1+1, c0:c1+1] += 1
                    h, s = layer["hollow"], layer["solid"]
                    directions.add((int(np.sign(s["top"]-h["top"])), int(np.sign(s["left"]-h["left"]))))
                overlap_counts.add(int(union.max()))
    assert len(signatures) == 42
    assert len(dimensions) >= 50 and len(layouts) >= 190
    assert backgrounds == set(range(10)) and sizes == set(range(3, 7))
    assert len(counts) >= 15 and len(rectangle_sizes) >= 70
    assert repeat_pairs >= 30 and edge_pairs >= 15
    assert {-1, 1} <= {d[0] for d in directions} and {-1, 1} <= {d[1] for d in directions}
    assert max(overlap_counts) >= 3
    # The fallback is also tested directly, so it does not rely on being hit
    # by an ordinary seed in the finite sweep.
    fallback_core = {"greedy", "horizontal", "horizontal_reverse", "vertical_reverse",
                     "nearest", "priority", "opaque_hole", "occlusion", "completion", "extend_tail"}
    for size in (3, 4):
        for counts in ((2, 3), (3, 2)):
            for color in range(10):
                for kind in ("H", "S"):
                    pair, a = _teaching_fallback(np.random.default_rng(2026+color), size, counts, color, kind)
                    assert fallback_core <= _witnesses(a)
                    assert np.array_equal(solve(pair["input"]), pair["output"])
    for size, counts in ((3, (3, 4)), (4, (4, 4)), (5, (3, 6)),
                         (5, (6, 3)), (6, (3, 4)), (6, (1, 1))):
        pair = _ordinary_fallback(size, counts, set())
        a = _parse(pair["input"])
        assert np.array_equal(solve(pair["input"]), pair["output"])
        if min(counts) > 2:
            assert "novel_depth" in _witnesses(a)
        if counts[0] != counts[1]:
            assert {"extend_tail", "inactive_"+("H" if counts[0] > counts[1] else "S")} <= _witnesses(a)
    for train, test in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(314159, train, test), generate_puzzle(314159, train, test)
        assert all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split])
                   for k in ("input", "output"))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Larger size, opposite-class top tie and nonblack background are valid.
    grid = np.full((11, 20), 7, dtype=np.int8)
    grid[1:9, 1:9] = _tile(8, 0, 7, "H")
    grid[1:9, 11:19] = _tile(8, 4, 7, "S")
    answer = solve(grid)
    assert np.all(answer[1:9, 9:11] == 4)
    assert np.all(answer[2:8, 2:8] == 7)
    grid[1, 2] = 7  # broken hollow rim
    try:
        solve(grid)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken stamp accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds")
