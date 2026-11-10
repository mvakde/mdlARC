"""Synthetic ARC tasks for center-expanded, transparent template panels."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    ((14, 25), "f694f2b631691d484cbfae6bb3499b5e8e591495cc05130ddc5716cb2b100ab3"),
    ((11, 22), "3b429586e4a7e8f9513d67a64b1bf7a8b46303eefb4c18503e2f4d6043840bc5"),
    ((12, 26), "c276890cc47f4215207fb19d74ec07f8bcb797736b77016801a30f6266d1a395"),
}


def _count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _resolve_counts(seed, num_train, num_test):
    _count(num_train, "num_train", 2, 10)
    _count(num_test, "num_test", 1, 4)
    code = int(seed)
    if num_train is None and num_test is None:
        num_test = 1 + code % 4
        num_train = num_test + 1 + (code // 4) % (10 - num_test)
    elif num_train is None:
        num_test = int(num_test)
        num_train = num_test + 1 + code % (10 - num_test)
    elif num_test is None:
        num_train = int(num_train)
        num_test = 1 + code % min(4, num_train - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _intervals(width, kind, middle_only=False):
    center = width // 2
    for x0 in range(1, width - 1):
        for x1 in range(x0, width - 1):
            if width % 2:
                role = "left" if x1 < center else "right" if x0 > center else "crossing"
            else:
                role = "left" if x1 < center else "right" if x0 >= center else "crossing"
            if role == kind and (not middle_only or x0 == x1 == center):
                yield x0, x1


def _project(obj, added):
    x0, x1 = obj["x0"], obj["x1"]
    if obj["kind"] == "right":
        return x0 + added, x1 + added
    if obj["kind"] == "crossing":
        return x0, x1 + added
    return x0, x1


def _overlap(a, b, added):
    ax0, ax1 = _project(a, added)
    bx0, bx1 = _project(b, added)
    return (max(a["y0"], b["y0"]) <= min(a["y1"], b["y1"])
            and max(ax0, bx0) <= min(ax1, bx1))


def _touch(a, b):
    yg = max(0, max(a["y0"], b["y0"]) - min(a["y1"], b["y1"]))
    xg = max(0, max(a["x0"], b["x0"]) - min(a["x1"], b["x1"]))
    return (yg == 0 and xg <= 1) or (xg == 0 and yg <= 1)


def _height_choice(rng, interior_height):
    return int(rng.integers(1, min(8, interior_height - 2) + 1))


def _place_one(rng, objects, width, height, added, kind, tag="", middle_only=False):
    intervals = list(_intervals(width, kind, middle_only))
    if not intervals:
        return False
    for attempt in range(140):
        if attempt < 110:
            x0, x1 = intervals[int(rng.integers(len(intervals)))]
            h = _height_choice(rng, height)
            y0 = int(rng.integers(1, height - h))
        else:
            x0, x1 = intervals[(attempt - 110) % len(intervals)]
            h = 1
            y0 = 1 + ((attempt - 110) // len(intervals)) % (height - 2)
        obj = dict(kind=kind, x0=x0, x1=x1, y0=y0, y1=y0 + h - 1, tag=tag)
        if not any(_overlap(obj, other, added) for other in objects):
            objects.append(obj)
            return True
    for y0 in range(1, height - 1):
        for x0, x1 in intervals:
            obj = dict(kind=kind, x0=x0, x1=x1, y0=y0, y1=y0, tag=tag)
            if not any(_overlap(obj, other, added) for other in objects):
                objects.append(obj)
                return True
    return False


def _place_pair(rng, objects, width, height, added, adjacent=False, same=False):
    lefts = list(_intervals(width, "left"))
    rights = list(_intervals(width, "right"))
    center = width // 2
    choices = [(l, r) for l in lefts for r in rights
               if (not adjacent or (width % 2 == 0 and l[1] == center - 1 and r[0] == center))
               and (not same or r[0] - l[1] >= 2)]
    if not choices:
        return False
    for attempt in range(160):
        l, r = choices[int(rng.integers(len(choices)))]
        h = _height_choice(rng, height)
        y0 = int(rng.integers(1, height - h))
        a = dict(kind="left", x0=l[0], x1=l[1], y0=y0, y1=y0 + h - 1, tag="pair_left")
        b = dict(kind="right", x0=r[0], x1=r[1], y0=y0, y1=y0 + h - 1,
                 tag="pair_same" if same else "pair_different")
        if not any(_overlap(q, old, added) for q in (a, b) for old in objects):
            objects.extend((a, b))
            return True
    for y0 in range(1, height - 1):
        for l, r in choices:
            a = dict(kind="left", x0=l[0], x1=l[1], y0=y0, y1=y0, tag="pair_left")
            b = dict(kind="right", x0=r[0], x1=r[1], y0=y0, y1=y0, tag="pair_same" if same else "pair_different")
            if not any(_overlap(q, old, added) for q in (a, b) for old in objects):
                objects.extend((a, b))
                return True
    return False


def _assign_colors(rng, objects, frame, template_bg, destination_bg):
    available = [c for c in range(10) if c not in (template_bg, destination_bg)]
    nonframe = [c for c in available if c != frame]
    # Assign frame bars first so subsequent touching pieces must differ.
    for obj in objects:
        if obj["tag"] == "frame_crossing":
            obj["color"] = frame
    for i, obj in enumerate(objects):
        if "color" in obj:
            continue
        forbidden = {other["color"] for other in objects if "color" in other and _touch(obj, other)}
        if obj["tag"] == "pair_same":
            color = objects[i - 1]["color"]
            if color in forbidden:
                return False
            obj["color"] = color
            continue
        if obj["tag"] == "pair_different":
            forbidden.add(objects[i - 1]["color"])
        if obj["tag"] == "pair_left" and i + 1 < len(objects) and objects[i + 1]["tag"] == "pair_same":
            forbidden |= {other["color"] for other in objects
                          if "color" in other and _touch(objects[i + 1], other)}
        options = [c for c in nonframe if c not in forbidden]
        if frame not in forbidden and (rng.random() < 0.14 or not options):
            obj["color"] = frame
            continue
        if not options:
            return False
        used = [other["color"] for other in objects if "color" in other and other["color"] in options]
        obj["color"] = int(rng.choice(used if used and rng.random() < 0.38 else options))
    return True


def _draw_pair(rng, spec):
    width = spec["width"]
    height = spec["height"]
    added = spec["added"]
    roles = [int(v) for v in rng.permutation(10)[:3]]
    frame, template_bg, destination_bg = roles
    objects = []
    paired = spec.get("paired", False)
    if paired:
        same = bool(spec.get("pair_same", False)) and not spec.get("adjacent", False)
        if not _place_pair(rng, objects, width, height, added, spec.get("adjacent", False), same):
            raise AssertionError("required side pair cannot fit")
    if spec.get("baseline", False) and not paired:
        for kind in ("left", "right"):
            if not _place_one(rng, objects, width, height, added, kind):
                raise AssertionError("baseline anchor cannot fit")
    if spec.get("frame_crossing", False):
        if not _place_one(rng, objects, width, height, added, "crossing", tag="frame_crossing"):
            raise AssertionError("frame crossing cannot fit")
    if spec.get("middle_only", False):
        if not _place_one(rng, objects, width, height, added, "crossing", middle_only=True):
            raise AssertionError("middle column cannot fit")
    if spec.get("baseline", False) and not any(o["kind"] == "crossing" for o in objects):
        if not _place_one(rng, objects, width, height, added, "crossing"):
            raise AssertionError("baseline crossing cannot fit")
    target_count = max(len(objects), int(spec["target_count"]))
    if not objects:
        kind = ("left", "right", "crossing")[int(rng.integers(3))]
        if not _place_one(rng, objects, width, height, added, kind):
            raise AssertionError("empty scene")
    misses = 0
    while len(objects) < target_count and misses < 12:
        kind = ("left", "right", "crossing")[int(rng.integers(3))]
        if _place_one(rng, objects, width, height, added, kind):
            misses = 0
        else:
            misses += 1
    if not _assign_colors(rng, objects, frame, template_bg, destination_bg):
        raise AssertionError("foreground coloring failed")
    template = np.full((height, width), template_bg, dtype=np.int8)
    destination = np.full((height, width + added), destination_bg, dtype=np.int8)
    for obj in objects:
        template[obj["y0"]:obj["y1"] + 1, obj["x0"]:obj["x1"] + 1] = obj["color"]
        x0, x1 = _project(obj, added)
        destination[obj["y0"]:obj["y1"] + 1, x0:x1 + 1] = obj["color"]
    total_width = 2 * width + added + 3
    inp = np.full((height + 2, total_width), frame, dtype=np.int8)
    divider = width + 1
    inp[1:-1, 1:divider] = template
    inp[1:-1, divider + 1:-1] = destination_bg
    out = inp.copy()
    out[1:-1, divider + 1:-1] = destination
    return {"input": inp, "output": out}


def _width(rng):
    return int(rng.integers(5, 12))


def _added(rng, width, source_like=False):
    maximum = 27 - 2 * width
    if source_like:
        return int(rng.integers(5, min(11, maximum) + 1))
    band = int(rng.integers(3))
    lo, hi = ((1, 2), (3, 5), (6, maximum))[band]
    if lo > maximum:
        lo, hi = 1, maximum
    return int(rng.integers(lo, min(hi, maximum) + 1))


def _make_spec(rng, width, source_like=False):
    return dict(width=width, height=int(rng.integers(7, 27)),
                added=_added(rng, width, source_like), target_count=int(rng.integers(1, 7)))


def _key(array):
    return array.shape, np.ascontiguousarray(array, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    source_like = rng.random() < 0.22
    train_widths = [6, 7] if source_like else [int(rng.choice([6, 8, 10])), int(rng.choice([5, 7, 9, 11]))]
    train_widths += [_width(rng) for _ in range(num_train - 2)]
    test_widths = [8 if source_like and rng.random() < 0.72 else _width(rng) for _ in range(num_test)]
    test_specs = []
    for width in test_widths:
        spec = _make_spec(rng, width, source_like and width == 8 and rng.random() < 0.5)
        spec["paired"] = bool(rng.random() < 0.44)
        spec["adjacent"] = bool(spec["paired"] and width % 2 == 0 and rng.random() < 0.40)
        spec["pair_same"] = bool(spec["paired"] and not spec["adjacent"] and rng.random() < 0.42)
        spec["frame_crossing"] = bool(rng.random() < 0.32)
        spec["middle_only"] = bool(width % 2 and rng.random() < 0.27)
        test_specs.append(spec)
    needs_pair = any(s["paired"] for s in test_specs)
    needs_adjacent = any(s["adjacent"] for s in test_specs)
    needs_frame = any(s["frame_crossing"] for s in test_specs)
    needs_middle = any(s["middle_only"] for s in test_specs)
    train_specs = [_make_spec(rng, w, source_like and i < 2 and rng.random() < 0.6)
                   for i, w in enumerate(train_widths)]
    first = train_specs[0]
    first["height"] = max(first["height"], 10)
    first["baseline"] = True
    first["paired"] = needs_pair or bool(rng.random() < 0.36)
    first["adjacent"] = needs_adjacent or bool(first["paired"] and rng.random() < 0.20)
    first["pair_same"] = bool(first["paired"] and not first["adjacent"] and rng.random() < 0.44)
    first["frame_crossing"] = needs_frame or bool(rng.random() < 0.25)
    second = train_specs[1]
    second["middle_only"] = needs_middle or bool(rng.random() < 0.25)
    for spec in train_specs[1:]:
        spec.setdefault("paired", bool(rng.random() < 0.32))
        spec["adjacent"] = bool(spec["paired"] and spec["width"] % 2 == 0 and rng.random() < 0.28)
        spec["pair_same"] = bool(spec["paired"] and not spec["adjacent"] and rng.random() < 0.42)
        spec["frame_crossing"] = bool(rng.random() < 0.23)
        spec.setdefault("middle_only", bool(spec["width"] % 2 and rng.random() < 0.18))
    if len({spec["added"] for spec in train_specs + test_specs}) == 1:
        second["added"] = 1 if second["added"] != 1 else 2
    # Each pair receives an independent stream; role colors are visible, pair-level choices.
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_train + num_test, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    used = set()
    demonstrated = set()
    for split, specs, offset in (("train", train_specs, 0), ("test", test_specs, num_train)):
        for i, spec in enumerate(specs):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + i]))
            for _ in range(24):
                try:
                    pair = _draw_pair(pair_rng, spec)
                except AssertionError:
                    continue
                pair_features, _ = _features(_inspect_input(pair["input"]))
                special = {"same_row_sides", "adjacent_cut", "frame_crossing", "middle_only"}
                if split == "test" and not (pair_features & special) <= demonstrated:
                    continue
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        demonstrated.update(pair_features)
                    break
            else:
                raise AssertionError("could not construct a unique valid scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(row, background):
    result = []
    x = 0
    while x < len(row):
        color = int(row[x])
        start = x
        while x + 1 < len(row) and int(row[x + 1]) == color:
            x += 1
        if color != background:
            result.append((start, x, color))
        x += 1
    return result


def _inspect_input(inp):
    h, w = inp.shape
    frame = int(inp[0, 0])
    assert np.all(inp[0] == frame) and np.all(inp[-1] == frame)
    assert np.all(inp[:, 0] == frame) and np.all(inp[:, -1] == frame)
    dividers = [x for x in range(1, w - 1) if np.all(inp[1:-1, x] == frame)]
    assert len(dividers) == 1
    divider = dividers[0]
    tw, dw = divider - 1, w - divider - 2
    assert 5 <= tw <= 11 and 1 <= dw - tw and tw + dw + 3 == w
    template = inp[1:-1, 1:divider]
    dest = inp[1:-1, divider + 1:-1]
    tb = int(template[0, 0])
    assert tb != frame and np.all(template[0] == tb) and np.all(template[-1] == tb)
    assert np.all(template[:, 0] == tb) and np.all(template[:, -1] == tb)
    db = int(dest[0, 0])
    assert db not in (frame, tb) and np.all(dest == db)
    assert np.all((template == tb) | (template != db))
    return dict(frame=frame, divider=divider, template=template, template_bg=tb,
                destination_bg=db, template_width=tw, destination_width=dw,
                added=dw - tw, height=h, width=w)


def _predict(input_grid):
    info = _inspect_input(input_grid)
    template = info["template"]
    width = info["template_width"]
    added = info["added"]
    rendered = np.full((len(template), info["destination_width"]), info["destination_bg"], dtype=np.int8)
    for y, row in enumerate(template):
        for x0, x1, color in _runs(row, info["template_bg"]):
            role = ("left" if x1 < width // 2 else "right" if x0 >= width // 2 else "crossing") if width % 2 == 0 else ("left" if x1 < width // 2 else "right" if x0 > width // 2 else "crossing")
            nx0 = x0 + added if role == "right" else x0
            nx1 = x1 if role == "left" else x1 + added
            assert np.all(rendered[y, nx0:nx1 + 1] == info["destination_bg"])
            rendered[y, nx0:nx1 + 1] = color
    output = input_grid.copy()
    output[1:-1, info["divider"] + 1:-1] = rendered
    return output


def _features(info):
    template, bg, width = info["template"], info["template_bg"], info["template_width"]
    features = set()
    lengths = []
    for row in template:
        runs = _runs(row, bg)
        roles = []
        for x0, x1, color in runs:
            role = ("left" if x1 < width // 2 else "right" if x0 >= width // 2 else "crossing") if width % 2 == 0 else ("left" if x1 < width // 2 else "right" if x0 > width // 2 else "crossing")
            roles.append((x0, x1, color, role))
            features.add(role)
            lengths.append(x1 - x0 + 1)
            if color == info["frame"]:
                features.add("frame_foreground")
                if role == "crossing":
                    features.add("frame_crossing")
            if width % 2 and x0 == x1 == width // 2:
                features.add("middle_only")
        for l in roles:
            for r in roles:
                if l[3] == "left" and r[3] == "right":
                    features.add("same_row_sides")
                    if width % 2 == 0 and l[1] == width // 2 - 1 and r[0] == width // 2 and l[2] != r[2]:
                        features.add("adjacent_cut")
                    if l[2] == r[2]:
                        features.add("same_color_sides")
                    else:
                        features.add("different_color_sides")
    return features, lengths


def _digest(pair):
    inp, out = pair["input"], pair["output"]
    input_bytes = np.ascontiguousarray(inp, dtype=np.int8).tobytes()
    output_bytes = np.ascontiguousarray(out, dtype=np.int8).tobytes()
    return inp.shape, hashlib.sha256(input_bytes + output_bytes).hexdigest()


def _foreground_components(template, background):
    height, width = template.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or int(template[y, x]) == background:
                continue
            color = int(template[y, x])
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and int(template[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            y0 = min(cy for cy, _ in cells)
            y1 = max(cy for cy, _ in cells)
            x0 = min(cx for _, cx in cells)
            x1 = max(cx for _, cx in cells)
            assert len(cells) == (y1 - y0 + 1) * (x1 - x0 + 1)
            components.append((color, y0, y1, x0, x1))
    return components


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _key(inp)
            pair_key = input_key, _key(out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _digest(pair) not in _SOURCE_DIGESTS
            info = _inspect_input(inp)
            assert np.array_equal(_predict(inp), out)
            assert np.array_equal(inp[:, :info["divider"] + 1], out[:, :info["divider"] + 1])
            assert 1 <= len(_foreground_components(info["template"], info["template_bg"])) <= 6
            info["features"], info["run_lengths"] = _features(info)
            infos[split].append(info)
    train_features = set().union(*(i["features"] for i in infos["train"]))
    assert {"left", "right", "crossing"} <= train_features
    assert len({i["added"] for split_infos in infos.values() for i in split_infos}) >= 2
    assert any(i["template_width"] % 2 == 0 for i in infos["train"])
    assert any(i["template_width"] % 2 == 1 for i in infos["train"])
    for info in infos["test"]:
        assert {"same_row_sides", "adjacent_cut", "frame_crossing", "middle_only"}.intersection(info["features"]) <= train_features


def self_test() -> None:
    a = generate_puzzle(101)
    b = generate_puzzle(202, 7, 3)
    assert all(_key(x["input"]) == _key(y["input"]) and _key(x["output"]) == _key(y["output"])
               for split in ("train", "test") for x, y in zip(b[split], generate_puzzle(202, 7, 3)[split]))
    for seed, nt, ne in ((401, 2, 1), (402, 10, 1), (403, 10, 4),
                         (501, None, 1), (502, None, 4), (503, 2, None), (504, 10, None)):
        p = generate_puzzle(seed, nt, ne)
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
        validate_generated_puzzle(p)
    for args in ((1, 1, None), (1, 11, None), (1, None, 0), (1, None, 5), (1, 2, 2), (1, 3, 4)):
        try:
            generate_puzzle(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    counts, shapes, layouts, colors, features, heights, additions, widths = (set() for _ in range(8))
    split_features = {"train": set(), "test": set()}
    component_counts, component_heights = set(), set()
    combined_features = set()
    within_bg_change = 0
    within_count_change = 0
    for seed in range(1000, 1042):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        counts.add((len(p["train"]), len(p["test"])))
        local_bgs = set()
        local_counts = set()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _inspect_input(pair["input"])
                feat, lens = _features(info)
                components = _foreground_components(info["template"], info["template_bg"])
                local_counts.add(len(components))
                component_counts.add(len(components))
                component_heights.update(c[2] - c[1] + 1 for c in components)
                features.update(feat)
                split_features[split].update(feat)
                combined_features.add(tuple(sorted(feat & {"same_row_sides", "frame_crossing", "middle_only", "adjacent_cut"})))
                widths.add(info["template_width"])
                additions.add(info["added"])
                heights.add(info["height"])
                shapes.add(pair["input"].shape)
                colors.add((info["frame"], info["template_bg"], info["destination_bg"]))
                local_bgs.add(info["template_bg"])
                layouts.add((info["height"], info["template_width"], info["added"],
                             tuple(tuple(_runs(row, info["template_bg"])) for row in info["template"])))
        within_bg_change += len(local_bgs) > 1
        within_count_change += len(local_counts) > 1
    assert len({a for a, b in counts}) >= 4 and len({b for a, b in counts}) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 150 and len(colors) >= 80
    assert widths == set(range(5, 12)) and 1 in additions and max(additions) >= 12
    assert min(heights) <= 10 and max(heights) >= 25 and within_bg_change >= 25
    assert {"same_row_sides", "adjacent_cut", "frame_crossing", "middle_only", "same_color_sides"} <= features
    assert all({"same_row_sides", "adjacent_cut", "frame_crossing", "middle_only"} <= split_features[split]
               for split in ("train", "test"))
    assert component_counts == set(range(1, 7)) and max(component_heights) >= 6
    assert within_count_change >= 25
    assert ("adjacent_cut", "frame_crossing", "same_row_sides") in combined_features
