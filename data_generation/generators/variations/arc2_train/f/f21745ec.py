"""Generate separated rectangular-stencil ARC puzzles."""

import hashlib

import numpy as np


# Exact source-pair fingerprints. Shapes alone are deliberately not excluded.
_SOURCE_PAIRS = {
    ((21, 21), "f25fe0454de7576bb355946faa252926f497d70b952a7a14516f8038ad292915",
     "aad54fdb75187dea94585e08fa72e2bd2aff3846a5cd00671eb298ce9b1cf325"),
    ((19, 19), "66850e70ccc69cd8b77cf9d0df5fb3b4b680d2b96f3941a03d131bd1474a6633",
     "f9acc0f62e043a6762dbfd0659c7151d7b1e1f894c0265f27ca6e615dbe47ba0"),
    ((22, 22), "de621f232dd8f345cfe2f3f1b5f8029cfd98a16dfb39ba0d757b5ebe37f36906",
     "fbf4556fa74b4280c47a2eaf856e91897c08f875153b7054d9f36820b82b7575"),
    ((23, 22), "dd5a61d14ffaaa46a586434df1de5f59a0e4cf66d83191a053ea430bfa6633a5",
     "5fd04d92a1026ad600be7e120fc25542c832267258d615d813e56d87076415a2"),
}


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    key = abs(int(seed))
    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        num_train = num_test + 1 + (key // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (key + 17) % (10 - num_test)
    elif num_test is None:
        num_test = 1 + (key + 23) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _frame(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _neighbors(cell, h, w):
    r, c = cell
    return [(a, b) for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= a < h and 0 <= b < w]


def _stencil(rng, h, w, mode, require_asymmetry):
    ih, iw = h - 2, w - 2
    area = ih * iw
    for _ in range(40):
        interior = np.zeros((ih, iw), dtype=bool)
        r = int(rng.integers(0, ih))
        c = int(rng.integers(0, iw))
        if mode == "sparse":
            density = float(rng.uniform(0.17, 0.62))
            interior = rng.random((ih, iw)) < density
        elif mode == "bar":
            horizontal = bool(rng.integers(0, 2))
            length = int(rng.integers(2, (iw if horizontal else ih) + 1))
            if horizontal:
                c = int(rng.integers(0, iw - length + 1))
                interior[r, c:c+length] = True
            else:
                r = int(rng.integers(0, ih - length + 1))
                interior[r:r+length, c] = True
        elif mode == "bend":
            vr = -1 if r >= ih // 2 else 1
            vc = -1 if c >= iw // 2 else 1
            vlen = min(ih, int(rng.integers(2, ih + 1)))
            hlen = min(iw, int(rng.integers(2, iw + 1)))
            for step in range(vlen):
                rr = r + vr * step
                if 0 <= rr < ih:
                    interior[rr, c] = True
            for step in range(hlen):
                cc = c + vc * step
                if 0 <= cc < iw:
                    interior[r, cc] = True
        elif mode == "zigzag":
            r, c = 0, int(rng.integers(0, iw))
            direction = 1 if c < iw // 2 else -1
            interior[r, c] = True
            for row in range(1, ih):
                interior[row, c] = True
                cc = c + direction
                if not 0 <= cc < iw:
                    direction *= -1
                    cc = c + direction
                interior[row, cc] = True
                c = cc
        else:
            target = int(rng.integers(2, max(3, int(area * 0.68) + 1)))
            occupied = {(r, c)}
            tip = (r, c)
            for _step in range(target - 1):
                if mode == "path":
                    frontier = [q for q in _neighbors(tip, ih, iw) if q not in occupied]
                    if not frontier:
                        frontier = [q for p in occupied for q in _neighbors(p, ih, iw)
                                    if q not in occupied]
                else:  # compact connected cluster
                    frontier = [q for p in occupied for q in _neighbors(p, ih, iw)
                                if q not in occupied]
                if not frontier:
                    break
                tip = frontier[int(rng.integers(0, len(frontier)))]
                occupied.add(tip)
            for rr, cc in occupied:
                interior[rr, cc] = True
        if bool(rng.integers(0, 2)):
            interior = np.flipud(interior)
        if bool(rng.integers(0, 2)):
            interior = np.fliplr(interior)
        count = int(interior.sum())
        asym = not any(np.array_equal(interior, other) for other in
                       (np.flipud(interior), np.fliplr(interior),
                        np.flipud(np.fliplr(interior))))
        if 1 <= count < area and (asym or not require_asymmetry):
            mask = _frame(h, w)
            mask[1:-1, 1:-1] = interior
            return mask
    # Every dimension sampled below has an interior of at least 3 by 3.
    mask = _frame(h, w)
    mask[1, 1] = mask[1, 2] = mask[2, 1] = True
    return mask


def _size(rng, exclude=()):
    # Independent aspect choice; all ordered sizes remain possible.
    category = int(rng.integers(0, 4))
    for _ in range(30):
        if category == 0:
            h = w = int(rng.integers(5, 10))
        elif category == 1:
            h, w = int(rng.integers(7, 10)), int(rng.integers(5, 7))
        elif category == 2:
            h, w = int(rng.integers(5, 7)), int(rng.integers(7, 10))
        else:
            h, w = (int(rng.integers(5, 10)) for _j in range(2))
        if (h, w) not in exclude:
            return h, w
    return next((h, w) for h in range(5, 10) for w in range(5, 10)
                if (h, w) not in exclude)


def _decoy_options(h, w, profile):
    options = []
    for dh in range(4, 11):
        for dw in range(4, 11):
            if (dh, dw) == (h, w):
                continue
            if profile == "same_h" and dh == h:
                options.append((dh, dw))
            elif profile == "same_w" and dw == w:
                options.append((dh, dw))
            elif profile == "both" and dh != h and dw != w:
                options.append((dh, dw))
            elif profile == "swapped" and h != w and (dh, dw) == (w, h):
                options.append((dh, dw))
            elif profile == "equal_area" and dh * dw == h * w:
                options.append((dh, dw))
            elif profile == "equal_perimeter" and dh + dw == h + w:
                options.append((dh, dw))
    return options


def _decoy(rng, h, w, profile):
    options = _decoy_options(h, w, profile)
    if not options:
        options = _decoy_options(h, w, "both")
    # Prefer close distractions without excluding distant ones.
    if bool(rng.integers(0, 4)):
        distance = min(abs(a-h) + abs(b-w) for a, b in options)
        options = [(a, b) for a, b in options if abs(a-h) + abs(b-w) <= distance + 2]
    return options[int(rng.integers(0, len(options)))]


def _place(rng, objects, allow_edge):
    ordered = sorted(objects, key=lambda o: o["h"] * o["w"], reverse=True)
    max_h, max_w = max(o["h"] for o in ordered), max(o["w"] for o in ordered)
    n = len(ordered)
    for attempt in range(36):
        floor = 16 + (2 if n >= 6 else 0) + (2 if n >= 8 else 0)
        H = int(rng.integers(max(floor, max_h + 2), 31))
        W = int(rng.integers(max(floor, max_w + 2), 31))
        if attempt >= 12:
            H = max(H, 25)
            W = max(W, 25)
        if attempt >= 25:
            H = W = 30
        placed = []
        for obj in ordered:
            h, w = obj["h"], obj["w"]
            valid = np.ones((H-h+1, W-w+1), dtype=bool)
            if not allow_edge:
                valid[0, :] = valid[-1, :] = False
                valid[:, 0] = valid[:, -1] = False
            for old in placed:
                # One complete background row or column separates boxes.
                r0 = max(0, old["r"] - h)
                r1 = min(H-h, old["r"] + old["h"])
                c0 = max(0, old["c"] - w)
                c1 = min(W-w, old["c"] + old["w"])
                valid[r0:r1+1, c0:c1+1] = False
            locations = np.argwhere(valid)
            if not len(locations):
                break
            # Bias some placements to an edge, while keeping all cells visible.
            edge = [p for p in locations if p[0] == 0 or p[1] == 0
                    or p[0] == H-h or p[1] == W-w] if allow_edge else []
            choices = edge if edge and rng.random() < 0.15 else locations
            r, c = choices[int(rng.integers(0, len(choices)))]
            placed.append(dict(obj, r=int(r), c=int(c)))
        if len(placed) == n:
            return placed, H, W
    raise ValueError("could not place separated rectangles")


def _pair(rng, h, w, recipients, decoys, profiles, colors, background, mode,
          asymmetric):
    stencil = _stencil(rng, h, w, mode, asymmetric)
    objects = [dict(kind="exemplar", h=h, w=w)]
    objects += [dict(kind="recipient", h=h, w=w) for _ in range(recipients)]
    for profile in profiles[:decoys]:
        dh, dw = _decoy(rng, h, w, profile)
        objects.append(dict(kind="decoy", h=dh, w=dw))
    while len(objects) < 1 + recipients + decoys:
        dh, dw = _decoy(rng, h, w, "both")
        objects.append(dict(kind="decoy", h=dh, w=dw))
    colors = [int(c) for c in colors if int(c) != background]
    color_count = int(rng.integers(2, min(len(objects), len(colors)) + 1))
    chosen = [int(x) for x in rng.choice(colors, size=color_count, replace=False)]
    assignments = [chosen[0], chosen[1]]
    assignments += [chosen[int(rng.integers(0, len(chosen)))]
                    for _ in range(len(objects)-2)]
    # Use every selected color, then scramble the unconstrained assignments.
    for i in range(2, min(len(chosen), len(objects))):
        assignments[i] = chosen[i]
    for obj, color in zip(objects, assignments):
        obj["color"] = color
    placed, H, W = _place(rng, objects, bool(rng.random() < 0.62))
    inp = np.full((H, W), background, dtype=np.int8)
    out = np.full_like(inp, background)
    for obj in placed:
        r, c, dh, dw = (obj[k] for k in ("r", "c", "h", "w"))
        mask = stencil if obj["kind"] == "exemplar" else _frame(dh, dw)
        inp[r:r+dh, c:c+dw][mask] = obj["color"]
        if (dh, dw) == (h, w):
            out[r:r+dh, c:c+dw][stencil] = obj["color"]
    return {"input": inp, "output": out}


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _source_key(pair):
    return (tuple(pair["input"].shape),
            hashlib.sha256(np.asarray(pair["input"], dtype=np.int8).tobytes()).hexdigest(),
            hashlib.sha256(np.asarray(pair["output"], dtype=np.int8).tobytes()).hexdigest())


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(int(seed) % (1 << 64))
    # A color vocabulary is sampled once. Color roles and backgrounds are visible
    # pair by pair; no hidden palette mapping is required to solve the task.
    vocabulary = [int(x) for x in rng.permutation(10)]
    backgrounds = vocabulary[:2]
    if bool(rng.integers(0, 2)):
        backgrounds.reverse()
    first_recipients = [int(x) for x in rng.choice([1, 2, 3, 4], size=2, replace=False)]
    first_size = _size(rng)
    second_size = _size(rng, (first_size,))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    modes = ("cluster", "path", "bar", "bend", "zigzag", "sparse")
    profiles = ("same_h", "same_w", "both", "swapped", "equal_area",
                "equal_perimeter")
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if split == "train" and index < 2:
                    h, w = (first_size, second_size)[index]
                    recipient_count = first_recipients[index]
                elif split == "test" and index == 0 and bool(vocabulary[2] % 2):
                    h, w = train_sizes[int(pair_rng.integers(0, len(train_sizes)))]
                    recipient_count = int(pair_rng.integers(1, 5))
                elif split == "test" and index == 0:
                    h, w = _size(pair_rng, train_sizes)
                    recipient_count = int(pair_rng.integers(1, 5))
                else:
                    h, w = _size(pair_rng)
                    recipient_count = int(pair_rng.integers(1, 5))
                decoy_count = int(pair_rng.integers(1, 4))
                if recipient_count >= 4 and decoy_count == 3 and max(h, w) >= 9:
                    decoy_count = 2
                if split == "train" and index < 2:
                    decoy_profile = profiles[index]
                    if index == 0 and decoy_count == 1:
                        decoy_count = 2
                    selected_profiles = [decoy_profile]
                else:
                    selected_profiles = []
                remaining = [p for p in profiles if p not in selected_profiles]
                pair_rng.shuffle(remaining)
                selected_profiles += remaining[:decoy_count-len(selected_profiles)]
                if split == "train" and index == 0:
                    mode = modes[int(pair_rng.integers(0, 5))]
                else:
                    mode = modes[int(pair_rng.integers(0, len(modes)))]
                bg = backgrounds[index] if split == "train" and index < 2 else (
                    vocabulary[int(pair_rng.integers(0, 10))])
                try:
                    pair = _pair(pair_rng, h, w, recipient_count, decoy_count,
                                 selected_profiles, vocabulary, bg, mode,
                                 split == "train" and index < 2)
                except ValueError:
                    continue
                key = _grid_key(pair["input"])
                if key in seen or _source_key(pair) in _SOURCE_PAIRS:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_sizes.append((h, w))
                break
            else:
                raise ValueError("could not construct a unique separated pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) > grid.size // 2
    foreground = grid != background
    candidates = []
    H, W = grid.shape
    for r, c in zip(*np.nonzero(foreground)):
        color = int(grid[r, c])
        if r and grid[r-1, c] == color:
            continue
        if c and grid[r, c-1] == color:
            continue
        w = 0
        while c+w < W and grid[r, c+w] == color:
            w += 1
        h = 0
        while r+h < H and grid[r+h, c] == color:
            h += 1
        if h < 4 or w < 4:
            continue
        sub = grid[r:r+h, c:c+w]
        if (np.all(sub[-1, :] == color) and np.all(sub[:, -1] == color)
                and np.all((sub == color) | (sub == background))):
            candidates.append(dict(r=int(r), c=int(c), h=h, w=w, color=color,
                                   mask=sub == color))
    # A decorated interior can contain another closed loop; the enclosing
    # frame remains the object.
    objects = [a for a in candidates if not any(
        a is not b and b["r"] <= a["r"] and b["c"] <= a["c"]
        and b["r"]+b["h"] >= a["r"]+a["h"]
        and b["c"]+b["w"] >= a["c"]+a["w"] for b in candidates)]
    assert 3 <= len(objects) <= 8
    covered = np.zeros(grid.shape, dtype=bool)
    for i, a in enumerate(objects):
        r, c, h, w = (a[k] for k in ("r", "c", "h", "w"))
        assert not np.any(covered[r:r+h, c:c+w])
        covered[r:r+h, c:c+w] = True
        assert np.all(a["mask"][0, :]) and np.all(a["mask"][-1, :])
        assert np.all(a["mask"][:, 0]) and np.all(a["mask"][:, -1])
        a["decorated"] = bool(a["mask"][1:-1, 1:-1].any())
        if a["decorated"]:
            assert np.any(~a["mask"][1:-1, 1:-1])
        for b in objects[i+1:]:
            assert (r+h < b["r"] or b["r"]+b["h"] < r
                    or c+w < b["c"] or b["c"]+b["w"] < c)
    assert np.array_equal(covered & foreground, foreground)
    exemplars = [a for a in objects if a["decorated"]]
    assert len(exemplars) == 1
    return background, objects, exemplars[0]


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    background, objects, exemplar = _analyze(inp)
    h, w = exemplar["h"], exemplar["w"]
    expected = np.full_like(inp, background)
    recipients, decoys = [], []
    for obj in objects:
        r, c = obj["r"], obj["c"]
        if (obj["h"], obj["w"]) == (h, w):
            expected[r:r+h, c:c+w][exemplar["mask"]] = obj["color"]
            if obj is not exemplar:
                recipients.append(obj)
                assert not obj["decorated"]
        else:
            decoys.append(obj)
            assert not obj["decorated"]
    assert 1 <= len(recipients) <= 4
    assert 1 <= len(decoys) <= 3
    assert any(obj["color"] != exemplar["color"] for obj in recipients)
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {"background": background, "objects": objects,
            "exemplar": exemplar, "recipients": recipients, "decoys": decoys}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            ikey = _grid_key(pair["input"])
            pkey = (ikey, _grid_key(pair["output"]))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            assert _source_key(pair) not in _SOURCE_PAIRS
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            analyses[split].append(_inspect_pair(pair))
    train = analyses["train"]
    assert len({(a["exemplar"]["h"], a["exemplar"]["w"]) for a in train}) >= 2
    assert len({a["background"] for a in train}) >= 2
    assert len({len(a["recipients"]) for a in train}) >= 2
    assert any(a["exemplar"]["mask"][1:-1, 1:-1].any() for a in train)
    assert any(any(d["h"] == a["exemplar"]["h"] for d in a["decoys"])
               for a in train)
    assert any(any(d["w"] == a["exemplar"]["w"] for d in a["decoys"])
               for a in train)
    assert any(not np.array_equal(a["exemplar"]["mask"][1:-1, 1:-1],
                                  np.fliplr(a["exemplar"]["mask"][1:-1, 1:-1]))
               for a in train)
    assert any(not np.array_equal(a["exemplar"]["mask"][1:-1, 1:-1],
                                  np.flipud(a["exemplar"]["mask"][1:-1, 1:-1]))
               for a in train)
    # Every rule needed by the test is visible: the exemplar is marked in
    # each input; training contrasts both dimensions and shows recoloring,
    # transfer, deletion, and top-left-relative orientation.


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(12345)
    assert _equal_puzzles(a, generate_puzzle(12345))
    assert not _equal_puzzles(a, generate_puzzle(12346))
    for tr, te in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(700+tr+te, tr, te)
        assert (len(p["train"]), len(p["test"])) == (tr, te)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(800+(tr or 0)+(te or 0), tr, te)
        assert (tr is None or len(p["train"]) == tr)
        assert (te is None or len(p["test"]) == te)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, layouts, stencils, sizes, counts, backgrounds = (
        set(), set(), set(), set(), set(), set())
    aspects, object_counts, frame_color_counts = set(), set(), set()
    profiles = set()
    split_profiles = {"train": set(), "test": set()}
    connected = sparse = edge = margin = repeated_color = unique_color = 0
    reused_test_size = novel_test_size = detailed_combination = bars = 0
    within_decoy_variation = within_object_variation = 0
    densities = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_decoys, puzzle_objects = set(), set()
        trained_sizes = {(a["exemplar"]["h"], a["exemplar"]["w"])
                         for a in (_inspect_pair(x) for x in puzzle["train"])}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _inspect_pair(pair)
                e = a["exemplar"]
                motif = e["mask"][1:-1, 1:-1]
                shapes.add(tuple(pair["input"].shape))
                layouts.add(tuple(sorted((x["r"], x["c"], x["h"], x["w"])
                                         for x in a["objects"])))
                stencils.add((e["h"], e["w"], motif.tobytes()))
                sizes.add((e["h"], e["w"]))
                counts.add((len(a["recipients"]), len(a["decoys"])))
                backgrounds.add(a["background"])
                aspects.add("square" if e["h"] == e["w"] else
                            "tall" if e["h"] > e["w"] else "wide")
                object_counts.add(len(a["objects"]))
                puzzle_objects.add(len(a["objects"]))
                puzzle_decoys.add(len(a["decoys"]))
                color_count = len({x["color"] for x in a["objects"]})
                frame_color_counts.add(color_count)
                reuse = color_count < len(a["objects"])
                repeated_color += reuse
                unique_color += not reuse
                touches_edge = any(x["r"] == 0 or x["c"] == 0 or
                                   x["r"]+x["h"] == pair["input"].shape[0] or
                                   x["c"]+x["w"] == pair["input"].shape[1]
                                   for x in a["objects"])
                edge += touches_edge
                margin += not touches_edge
                detailed_combination += touches_edge and reuse and int(motif.sum()) >= 5
                densities.append(float(motif.mean()))
                bars += (np.any(motif, axis=0).sum() == 1 or
                         np.any(motif, axis=1).sum() == 1)
                for d in a["decoys"]:
                    dh, dw = d["h"], d["w"]
                    if dh == e["h"]:
                        profiles.add("same_h")
                        split_profiles[split].add("same_h")
                    if dw == e["w"]:
                        profiles.add("same_w")
                        split_profiles[split].add("same_w")
                    if dh != e["h"] and dw != e["w"]:
                        profiles.add("both")
                        split_profiles[split].add("both")
                    if (dh, dw) == (e["w"], e["h"]) and e["h"] != e["w"]:
                        profiles.add("swapped")
                        split_profiles[split].add("swapped")
                    if dh * dw == e["h"] * e["w"]:
                        profiles.add("equal_area")
                        split_profiles[split].add("equal_area")
                    if dh + dw == e["h"] + e["w"]:
                        profiles.add("equal_perimeter")
                        split_profiles[split].add("equal_perimeter")
                remaining = set(zip(*np.nonzero(motif)))
                components = 0
                while remaining:
                    components += 1
                    frontier = [remaining.pop()]
                    while frontier:
                        cell = frontier.pop()
                        for neighbor in _neighbors(cell, *motif.shape):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                frontier.append(neighbor)
                connected += components == 1
                sparse += components > 1
                if split == "test":
                    reused_test_size += (e["h"], e["w"]) in trained_sizes
                    novel_test_size += (e["h"], e["w"]) not in trained_sizes
        within_decoy_variation += len(puzzle_decoys) >= 2
        within_object_variation += len(puzzle_objects) >= 2
    assert len(shapes) >= 20 and len(layouts) >= 80 and len(stencils) >= 80
    assert len(sizes) >= 12 and len(counts) >= 8 and len(backgrounds) >= 5
    assert aspects == {"square", "tall", "wide"}
    assert object_counts == {3, 4, 5, 6, 7, 8}
    assert {1, 2, 3, 4}.issubset({r for r, _ in counts})
    assert {1, 2, 3}.issubset({d for _, d in counts})
    assert len(frame_color_counts) >= 5
    assert profiles == {"same_h", "same_w", "both", "swapped",
                        "equal_area", "equal_perimeter"}
    assert all(len(value) >= 5 for value in split_profiles.values())
    assert connected >= 30 and sparse >= 10
    assert edge >= 15 and margin >= 15
    assert repeated_color >= 15 and unique_color >= 15
    assert detailed_combination >= 15 and bars >= 10
    assert min(densities) < 0.15 and max(densities) > 0.6
    assert within_decoy_variation >= 15 and within_object_variation >= 30
    assert reused_test_size >= 5 and novel_test_size >= 5
