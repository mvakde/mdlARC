"""Generate closed-frame marker-count puzzles with sorted unary outputs."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "dbb5dde4a91854bd1f6c1976f65a6e79d004bd45714a64c1a21711f91b87640a",
    "072fa83f1cd78607cb900ce1079602ba8b09de75bad5359a76467ab7b3bb1f15",
    "f175cb5eeba721b8cea5f95a490b8e9eb335da03f734fffe8dab9d689424c74b",
    "9ea3993cc4703f5de4eb7290cdec0856c129ad6900ede90b170c65a03368272e",
}
_LAYOUTS = ("horizontal", "vertical", "diagonal", "grid", "mixed")


def _optional_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _pair_counts(rng, num_train, num_test):
    num_train = _optional_count("num_train", num_train)
    num_test = _optional_count("num_test", num_test)
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    return num_train, num_test


def _slots(rng, n, layout):
    """Allocate nonoverlapping rectangular cells, then vary boxes inside them."""
    gap = int(rng.integers(1, 3)) if n < 5 else 1
    lead = int(rng.integers(0, 3)) if n < 5 else int(rng.integers(0, 2))
    cells = []
    if layout == "horizontal":
        upper = (29 - lead - gap * (n - 1)) // n
        w = int(rng.integers(4, upper + 1))
        h = int(rng.integers(6, 20))
        for i in range(n):
            cells.append((lead + i * (w + gap), int(rng.integers(0, 5)), w, h))
    elif layout == "vertical":
        upper = (29 - lead - gap * (n - 1)) // n
        h = int(rng.integers(4, upper + 1))
        w = int(rng.integers(6, 20))
        for i in range(n):
            cells.append((int(rng.integers(0, 5)), lead + i * (h + gap), w, h))
    elif layout == "diagonal":
        upper = (29 - lead - gap * (n - 1)) // n
        side = int(rng.integers(5, upper + 1))
        for i in range(n):
            cells.append((lead + i * (side + gap), lead + i * (side + gap), side, side))
    elif layout == "grid":
        columns = 2 if n <= 4 else int(rng.integers(2, 4))
        rows = (n + columns - 1) // columns
        max_w = (29 - lead - gap * (columns - 1)) // columns
        max_h = (29 - lead - gap * (rows - 1)) // rows
        w = int(rng.integers(5, max_w + 1))
        h = int(rng.integers(5, max_h + 1))
        positions = [(i // columns, i % columns) for i in range(n)]
        rng.shuffle(positions)
        for row, col in positions:
            cells.append((lead + col * (w + gap), lead + row * (h + gap), w, h))
    elif layout == "mixed":
        left_n = int(rng.integers(1, n))
        right_n = n - left_n
        left_w = int(rng.integers(5, 13))
        right_w = int(rng.integers(5, min(13, 29 - lead - gap - left_w) + 1))
        side_specs = [(left_n, lead, left_w), (right_n, lead + left_w + gap, right_w)]
        if rng.integers(2):
            side_specs.reverse()
        for side_n, x, w in side_specs:
            cell_h = (28 - gap * (side_n - 1)) // side_n
            cell_h = min(cell_h, 21)
            for j in range(side_n):
                cells.append((x, lead + j * (cell_h + gap), w, cell_h))
        rng.shuffle(cells)
    elif layout == "source_like":
        left_w = int(rng.integers(5, 11))
        right_w = int(rng.integers(5, 12))
        top_h = int(rng.integers(5, 9))
        bottom_h = int(rng.integers(5, 9))
        middle_gap = int(rng.integers(1, 4))
        x = lead + left_w + gap
        cells = [
            (lead, lead, left_w, top_h),
            (lead, lead + top_h + middle_gap, left_w, bottom_h),
            (x, lead, right_w, top_h + middle_gap + bottom_h),
        ]
    else:
        raise ValueError("unknown layout")
    return cells


def _sample_box(rng, cap_w, cap_h, count, density, min_h=4):
    sides = []
    for top in range(1, 5):
        for bottom in range(1, 5):
            ch = cap_h - top - bottom
            if ch < 2:
                continue
            for left in range(1, 5):
                for right in range(1, 5):
                    cw = cap_w - left - right
                    if cw >= 2 and ch * cw > count:
                        sides.append((top, bottom, left, right))
    if not sides:
        raise ValueError("slot cannot hold the requested count")
    top, bottom, left, right = sides[int(rng.integers(len(sides)))]
    possibilities = [
        (ch, cw) for ch in range(2, cap_h - top - bottom + 1)
        for cw in range(2, cap_w - left - right + 1)
        if ch * cw > count and top + ch + bottom >= min_h
    ]
    if not possibilities:
        raise ValueError("slot cannot meet the minimum frame height")
    if density == "dense":
        possibilities.sort(key=lambda p: (p[0] * p[1], abs(p[0] - p[1])))
        possibilities = possibilities[:max(1, len(possibilities) // 3)]
    elif density == "sparse":
        possibilities.sort(key=lambda p: p[0] * p[1], reverse=True)
        possibilities = possibilities[:max(1, len(possibilities) // 3)]
    ch, cw = possibilities[int(rng.integers(len(possibilities)))]
    return {"top": top, "bottom": bottom, "left": left, "right": right,
            "ch": ch, "cw": cw, "h": top + ch + bottom, "w": left + cw + right}


def _marker_positions(rng, h, w, count, mode):
    positions = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(positions)
    if mode == "clump":
        focus = (int(rng.integers(h)), int(rng.integers(w)))
        positions.sort(key=lambda p: abs(p[0] - focus[0]) + abs(p[1] - focus[1]))
    elif mode == "edge":
        positions.sort(key=lambda p: min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]))
    elif mode == "isolated":
        chosen = []
        for p in positions:
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in chosen):
                chosen.append(p)
            if len(chosen) == count:
                break
        positions = chosen + [p for p in positions if p not in chosen]
    return positions[:count]


def _make_pair(rng, n, layout, colors, forced_evidence=False):
    background, marker, palette = colors
    slots = _slots(rng, n, layout)
    capacities = [(w - 2) * (h - 2) - 1 for _, _, w, h in slots]
    maximum = min(15, min(capacities))
    if maximum < n:
        raise ValueError("insufficient distinct counts")
    difficulty = int(rng.integers(3))
    if difficulty == 0:
        ceiling = min(maximum, max(n, int(rng.integers(n, 7))))
    elif difficulty == 1:
        ceiling = min(maximum, max(n, int(rng.integers(6, 11))))
    else:
        ceiling = maximum
    counts = [int(v) for v in rng.choice(np.arange(1, ceiling + 1), size=n, replace=False)]
    frame_colors = [int(v) for v in rng.choice(palette, size=n, replace=False)]
    if forced_evidence:
        # A three-row example refutes literal color order and spatial order.
        for _ in range(30):
            if frame_colors != sorted(frame_colors) and frame_colors != sorted(frame_colors, reverse=True):
                break
            rng.shuffle(frame_colors)
        if frame_colors == sorted(frame_colors) or frame_colors == sorted(frame_colors, reverse=True):
            frame_colors = [frame_colors[1], frame_colors[2], frame_colors[0]]
    objects = []
    for i, (x0, y0, cap_w, cap_h) in enumerate(slots):
        density = ("dense", "sparse", "ordinary")[int(rng.integers(3))]
        min_h = max(obj["y"] + obj["h"] for obj in objects) - y0 if layout == "source_like" and i == 2 else 4
        box = _sample_box(rng, cap_w, cap_h, counts[i], density, min_h)
        x = x0 if layout == "source_like" else x0 + int(rng.integers(cap_w - box["w"] + 1))
        y = y0 if layout == "source_like" and i == 2 else y0 + int(rng.integers(cap_h - box["h"] + 1))
        objects.append({**box, "x": x, "y": y, "color": frame_colors[i],
                        "count": counts[i], "mode": ("scatter", "clump", "edge", "isolated")[int(rng.integers(4))]})
    if forced_evidence:
        for _ in range(30):
            ordered = sorted(objects, key=lambda q: (q["y"], q["x"]))
            if [q["count"] for q in ordered] != sorted(counts):
                break
            perm = rng.permutation(n)
            recolor = [objects[int(i)]["color"] for i in perm]
            recount = [objects[int(i)]["count"] for i in perm]
            for obj, color, count in zip(objects, recolor, recount):
                obj["color"], obj["count"] = color, count
        # Color order is assessed in count order, after reassignment.
        for _ in range(30):
            order = [q["color"] for q in sorted(objects, key=lambda q: q["count"])]
            if order != sorted(order) and order != sorted(order, reverse=True):
                break
            shuffled = [q["color"] for q in objects]
            rng.shuffle(shuffled)
            for obj, color in zip(objects, shuffled):
                obj["color"] = color
    if any(obj["x"] == 0 and obj["y"] == 0 for obj in objects):
        for obj in objects:
            obj["y"] += 1
    required_h = max(obj["y"] + obj["h"] for obj in objects)
    required_w = max(obj["x"] + obj["w"] for obj in objects)
    h = min(30, required_h + int(rng.integers(0, 5)))
    w = min(30, required_w + int(rng.integers(0, 5)))
    grid = np.full((h, w), background, dtype=np.int8)
    for obj in objects:
        y, x = obj["y"], obj["x"]
        grid[y:y + obj["h"], x:x + obj["w"]] = obj["color"]
        cy, cx = y + obj["top"], x + obj["left"]
        grid[cy:cy + obj["ch"], cx:cx + obj["cw"]] = background
        for dy, dx in _marker_positions(rng, obj["ch"], obj["cw"], obj["count"], obj["mode"]):
            grid[cy + dy, cx + dx] = marker
    ordered = sorted(objects, key=lambda q: q["count"])
    output = np.full((n, ordered[-1]["count"]), background, dtype=np.int8)
    for row, obj in enumerate(ordered):
        output[row, :obj["count"]] = obj["color"]
    return {"input": grid, "output": output}


def _signature(array):
    return (tuple(array.shape), array.tobytes())


def _frame(grid, color):
    points = np.argwhere(grid == color)
    if len(points) < 12:
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    local = grid[y0:y1 + 1, x0:x1 + 1] == color
    hole = np.argwhere(~local)
    if not len(hole):
        return None
    hy0, hx0 = hole.min(axis=0)
    hy1, hx1 = hole.max(axis=0)
    if min(hy0, hx0, local.shape[0] - 1 - hy1, local.shape[1] - 1 - hx1) < 1:
        return None
    if max(hy0, hx0, local.shape[0] - 1 - hy1, local.shape[1] - 1 - hx1) > 4:
        return None
    if hy1 - hy0 + 1 < 2 or hx1 - hx0 + 1 < 2:
        return None
    expected = np.ones(local.shape, dtype=bool)
    expected[hy0:hy1 + 1, hx0:hx1 + 1] = False
    if not np.array_equal(local, expected):
        return None
    return {"color": int(color), "y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1),
            "cy0": int(y0 + hy0), "cy1": int(y0 + hy1), "cx0": int(x0 + hx0), "cx1": int(x0 + hx1),
            "sides": (int(hy0), int(local.shape[0] - 1 - hy1), int(hx0), int(local.shape[1] - 1 - hx1))}


def _analyze_input(grid):
    background = int(grid[0, 0])
    frame_colors = []
    frames = []
    for color in (int(v) for v in np.unique(grid) if int(v) != background):
        frame = _frame(grid, color)
        if frame is not None:
            frame_colors.append(color)
            frames.append(frame)
    assert 2 <= len(frames) <= 5
    residual = set(int(v) for v in np.unique(grid)) - {background} - set(frame_colors)
    assert len(residual) == 1
    marker = residual.pop()
    occupied = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        y0, y1, x0, x1 = frame["y0"], frame["y1"], frame["x0"], frame["x1"]
        assert not np.any(occupied[y0:y1 + 1, x0:x1 + 1])
        occupied[y0:y1 + 1, x0:x1 + 1] = True
        cavity = grid[frame["cy0"]:frame["cy1"] + 1, frame["cx0"]:frame["cx1"] + 1]
        assert np.all((cavity == background) | (cavity == marker))
        frame["count"] = int(np.count_nonzero(cavity == marker))
        frame["area"] = int(cavity.size)
        assert 1 <= frame["count"] < frame["area"]
    assert np.all(occupied[grid == marker])
    assert len({f["count"] for f in frames}) == len(frames)
    return background, marker, frames


def _predicted_output(grid):
    background, marker, frames = _analyze_input(grid)
    frames = sorted(frames, key=lambda f: f["count"])
    output = np.full((len(frames), frames[-1]["count"]), background, dtype=np.int8)
    for row, frame in enumerate(frames):
        output[row, :frame["count"]] = frame["color"]
    return output


def _layout_kind(frames):
    x_clear = all(a["x1"] < b["x0"] or b["x1"] < a["x0"] for i, a in enumerate(frames) for b in frames[i + 1:])
    y_clear = all(a["y1"] < b["y0"] or b["y1"] < a["y0"] for i, a in enumerate(frames) for b in frames[i + 1:])
    if x_clear and y_clear:
        return "diagonal"
    if x_clear:
        return "horizontal"
    if y_clear:
        return "vertical"
    return "mixed"


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    base_colors = [int(v) for v in rng.permutation(10)]
    stable_roles = bool(rng.integers(2))
    first_layout = ("horizontal", "vertical")[int(rng.integers(2))]
    second_layout = "vertical" if first_layout == "horizontal" else "horizontal"
    seen_inputs, seen_test_outputs = set(), set()
    train, test = [], []
    for split, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            if split == "train" and index == 0:
                n, layout = int(rng.integers(3, 5)), first_layout
            elif split == "train" and index == 1:
                n, layout = int(rng.integers(2, 5)), second_layout
            else:
                n = int(rng.choice([2, 3, 4, 5], p=[0.23, 0.37, 0.30, 0.10]))
                if split == "test" and n == 3 and rng.random() < 0.20:
                    layout = "source_like"
                else:
                    allowed = _LAYOUTS if n <= 4 else tuple(v for v in _LAYOUTS if v != "diagonal")
                    layout = allowed[int(rng.integers(len(allowed)))]
            for attempt in range(80):
                if stable_roles or (split == "train" and index == 0):
                    colors = (base_colors[0], base_colors[1], base_colors[2:])
                else:
                    perm = [int(v) for v in rng.permutation(10)]
                    if split == "train" and index == 1 and (perm[0] == base_colors[0] or perm[1] == base_colors[1]):
                        continue
                    colors = (perm[0], perm[1], perm[2:])
                try:
                    pair = _make_pair(rng, n, layout, colors, split == "train" and index == 0)
                    assert np.array_equal(_predicted_output(pair["input"]), pair["output"])
                    if split == "train" and index == 0:
                        first_frames = _analyze_input(pair["input"])[2]
                        if not any(
                            a["count"] < b["count"] and a["area"] > b["area"]
                            for a in first_frames for b in first_frames
                        ):
                            continue
                    input_key = _signature(pair["input"])
                    output_key = _signature(pair["output"])
                    if input_key in seen_inputs or (split == "test" and output_key in seen_test_outputs):
                        continue
                    seen_inputs.add(input_key)
                    if split == "test":
                        seen_test_outputs.add(output_key)
                    (train if split == "train" else test).append(pair)
                    break
                except (ValueError, AssertionError):
                    continue
            else:
                raise ValueError("could not construct an unambiguous unique pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys, test_outputs = set(), set(), set()
    layouts, role_pairs = set(), []
    spatial_conflict = color_conflict = area_conflict = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert not (inp.shape == out.shape and np.array_equal(inp, out))
            ikey, okey = _signature(inp), _signature(out)
            assert ikey not in input_keys and (ikey, okey) not in pair_keys
            input_keys.add(ikey)
            pair_keys.add((ikey, okey))
            digest = hashlib.sha256(inp.astype(np.uint8).tobytes() + b"|" + out.astype(np.uint8).tobytes()).hexdigest()
            assert digest not in _SOURCE_PAIR_DIGESTS
            background, marker, frames = _analyze_input(inp)
            assert np.array_equal(_predicted_output(inp), out)
            role_pairs.append((background, marker))
            if split == "train":
                layouts.add(_layout_kind(frames))
                area_conflict |= any(
                    a["count"] < b["count"] and a["area"] > b["area"]
                    for a in frames for b in frames
                )
                spatial = [f["count"] for f in sorted(frames, key=lambda f: (f["y0"], f["x0"]))]
                spatial_conflict |= spatial != sorted(spatial)
                ordered_colors = [f["color"] for f in sorted(frames, key=lambda f: f["count"])]
                color_conflict |= ordered_colors != sorted(ordered_colors) and ordered_colors != sorted(ordered_colors, reverse=True)
            else:
                assert okey not in test_outputs
                test_outputs.add(okey)
    assert len(layouts) >= 2
    assert spatial_conflict and color_conflict and area_conflict
    # The only cross-pair convention is the count-and-sort rule. Colors are
    # self-identifying in every input, so stable and changing palettes coexist.
    assert all(background != marker for background, marker in role_pairs)


def _puzzles_equal(a, b):
    return all(np.array_equal(pa[k], pb[k]) for split in ("train", "test")
               for pa, pb in zip(a[split], b[split]) for k in ("input", "output")) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(918273, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(918273, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(918274, 7, 3))
    for seed, tr, te in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        validate_generated_puzzle(puzzle)
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (2.5, 1), (True, 1), (3, False)):
        try:
            generate_puzzle(3001, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    train_counts, test_counts, shapes, structures, roles, kinds = set(), set(), set(), set(), set(), set()
    all_sides, all_counts, all_frame_n = set(), set(), set()
    varied_roles = novel_test_counts = new_test_colors = 0
    edge_pairs = thick_pairs = narrow_pairs = wide_pairs = 0
    touching_frames = isolated_frames = 0
    left_max_tests = top_max_tests = smaller_max_tests = 0
    inverted_density_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_roles = set()
        demonstrated_counts = set()
        demonstrated_colors = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, marker, frames = _analyze_input(inp)
                pair_roles.add((bg, marker))
                roles.add((bg, marker))
                shapes.add(tuple(inp.shape))
                kinds.add(_layout_kind(frames))
                all_frame_n.add(len(frames))
                all_counts.update(f["count"] for f in frames)
                all_sides.update(s for f in frames for s in f["sides"])
                structures.add((tuple(inp.shape), tuple(sorted((f["x1"] - f["x0"], f["y1"] - f["y0"], f["area"], f["count"], f["sides"]) for f in frames))))
                assert np.array_equal(_predicted_output(inp.copy()), pair["output"])
                if split == "train":
                    demonstrated_counts.update(f["count"] for f in frames)
                    demonstrated_colors.update(f["color"] for f in frames)
                else:
                    novel_test_counts += any(f["count"] not in demonstrated_counts for f in frames)
                    new_test_colors += any(f["color"] not in demonstrated_colors for f in frames)
                    maximum = max(frames, key=lambda f: f["count"])
                    left_max_tests += maximum is min(frames, key=lambda f: f["x0"])
                    top_max_tests += maximum is min(frames, key=lambda f: f["y0"])
                    smaller_max_tests += maximum is not max(frames, key=lambda f: f["area"])
                edge_pairs += any(f["x0"] == 0 or f["y0"] == 0 or f["x1"] == inp.shape[1] - 1 or f["y1"] == inp.shape[0] - 1 for f in frames)
                thick_pairs += any(max(f["sides"]) >= 3 for f in frames)
                narrow_pairs += pair["output"].shape[1] <= 3
                wide_pairs += pair["output"].shape[1] >= 9
                inverted_density_pairs += any(a["count"] < b["count"] and a["area"] > b["area"] for a in frames for b in frames)
                for frame in frames:
                    cavity = inp[frame["cy0"]:frame["cy1"] + 1, frame["cx0"]:frame["cx1"] + 1] == marker
                    touches = bool(np.any(cavity[1:] & cavity[:-1]) or np.any(cavity[:, 1:] & cavity[:, :-1]))
                    touching_frames += touches
                    isolated_frames += not touches and frame["count"] >= 2
        varied_roles += len(pair_roles) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(structures) >= 100 and len(roles) >= 18
    assert {2, 3, 4, 5}.issubset(all_frame_n)
    assert {1, 2, 3, 4}.issubset(all_sides)
    assert min(all_counts) == 1 and max(all_counts) >= 9
    assert {"horizontal", "vertical", "mixed"}.issubset(kinds)
    assert varied_roles >= 8 and novel_test_counts >= 8 and new_test_colors >= 5
    assert edge_pairs >= 30 and thick_pairs >= 30
    assert narrow_pairs >= 5 and wide_pairs >= 20
    assert touching_frames >= 50 and isolated_frames >= 20
    assert left_max_tests >= 10 and top_max_tests >= 10 and smaller_max_tests >= 10
    assert inverted_density_pairs >= 30
